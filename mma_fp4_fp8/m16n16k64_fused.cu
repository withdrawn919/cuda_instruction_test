#include <cstdint>
#include <cuda_runtime.h>
#include <cuda.h>
#include <mma.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>
#include <cstdio>
#include <iostream>
#include "../dequantize_fp4_fp8/wmma_utils.cuh"
// #include "../dequantize_fp4_fp8/wmma.cuh"


using packed_act_t = uint4;
using packed_wgt_t = uint4;
using e4m3 = __nv_fp8_e4m3;
using e2m1 = __nv_fp4_e2m1;
using fp8 = __nv_fp8_e4m3;

// 16 * 16 matrix
struct alignas(32) packed_f32psum_t {
    float data[8];
};
namespace m16n16k64{
    constexpr int M = 16;
    constexpr int N = 16;  
    constexpr int K = 64;
    constexpr int WARP_SIZE = 32;
}
struct packed_float4_t{
    float data[4];
};
#define CHECK_CUDA(status)                                              \
  {                                                                     \
    cudaError_t error = status;                                         \
    if (error != cudaSuccess) {                                         \
      std::cerr << "Got bad cuda status: " << cudaGetErrorString(error) \
                << " at line: " << __LINE__ << std::endl;               \
      exit(EXIT_FAILURE);                                               \
    }                                                                   \
  }

  #define HMMA16816(RD0, RD1, RA0, RA1, RB0, RC0, RC1)     \
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.e4m3.e4m3.f16   \
                 {%0, %1},                                              \
                 {%2, %3},                                              \
                 {%4},                                                 \
                 {%5, %6};\n"                                           \
                 : "=r"(RD0), "=r"(RD1)                                 \
                 : "r"(RA0), "r"(RA1),                                  \
                   "r"(RB0),                                            \
                   "r"(RC0), "r"(RC1))

  struct SM75_16x8x8_F32F16F16F32_TN
  {
    using DRegisters = float[4];
    using ARegisters = uint32_t[2];
    using BRegisters = uint32_t[1];
    using CRegisters = float[4];
  
    // Register asm fma
    __device__ static void
    fma(float         & d0, float         & d1, float      & d2, float      & d3,
      uint32_t const& a0, uint32_t const& a1,
      uint32_t const& b0,
      float    const& c0, float    const& c1, float const& c2, float const& c3)
    {
      asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"
          "{%0, %1, %2, %3},"
          "{%4, %5},"
          "{%6},"
          "{%7, %8, %9, %10};\n"
          : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
          :  "r"(a0),  "r"(a1),
             "r"(b0),
             "f"(c0),  "f"(c1),  "f"(c2),  "f"(c3));
  
    }
  
    __device__ static void
    load_B_fp8_G(const fp8 *B,BRegisters& b_regs)
    {
      const int laneid = threadIdx.x % 32;
      const int index_16_8_8_b = laneid % 4 * 2 * 8 + laneid / 4;
      fp8 B0 = B[index_16_8_8_b];
      fp8 B1 = B[index_16_8_8_b + 8];

      __half2 half2_B0_B1 = __half2{static_cast<half>(B0), static_cast<half>(B1)};

      b_regs[0] = *reinterpret_cast<uint32_t*>(&half2_B0_B1);
    }
  };
  __device__ packed_float4_t ptx_m16n8k16_m16n8k8(fp8* input_A, float* input_B, fp8* input_C) {
    
    //---------------- 第一个tensor core计算开始------------
    const size_t laneid = threadIdx.x % 32;

    uint32_t RA[2]{0};
    uint32_t RB[1]{0};
    uint32_t RC[2]{0};
    uint32_t RD[2]{0};
    
    int index_a = laneid / 4 * 16 + laneid % 4 * 4;
    int index_b = laneid % 4 * 8 * 4 + laneid /4;

    RA[0] = *(uint32_t*)&input_A[index_a];
    RA[1] = *(uint32_t*)&input_A[index_a + 16 * 8];
    fp8 RB0 = static_cast<fp8>(input_B[index_b]);
    fp8 RB1 = static_cast<fp8>(input_B[index_b + 8]);
    fp8 RB2 = static_cast<fp8>(input_B[index_b + 16]);
    fp8 RB3 = static_cast<fp8>(input_B[index_b + 24]);
    RB[0] |= static_cast<uint32_t>(RB0.__x);
    RB[0] |= static_cast<uint32_t>(RB1.__x) << 8;
    RB[0] |= static_cast<uint32_t>(RB2.__x) << 16;
    RB[0] |= static_cast<uint32_t>(RB3.__x) << 24;
    //执行mma执行
    HMMA16816(RD[0], RD[1],
              RA[0], RA[1],
              RB[0],
              RC[0], RC[1]); 
    //---------------- 第一个tensor core计算结束------------
    //---------------- 第二个tensor core计算开始------------        
    SM75_16x8x8_F32F16F16F32_TN::ARegisters fragment_A;
    // load A
    fragment_A[0] = RD[0];
    fragment_A[1] = RD[1];
    // load B
    SM75_16x8x8_F32F16F16F32_TN::BRegisters fragment_B = {0};
    SM75_16x8x8_F32F16F16F32_TN::load_B_fp8_G(input_C, fragment_B);
    // initialize C
    SM75_16x8x8_F32F16F16F32_TN::CRegisters fragment_C = {0.0f, 0.0f, 0.0f, 0.0f};
    // initialize D
    SM75_16x8x8_F32F16F16F32_TN::DRegisters fragment_D = {0.0f, 0.0f, 0.0f, 0.0f};
    // execute the MMA operation
    SM75_16x8x8_F32F16F16F32_TN::fma(fragment_D[0], fragment_D[1], fragment_D[2], fragment_D[3],
        RD[0], RD[1],
        fragment_B[0],
        fragment_C[0], fragment_C[1], fragment_C[2], fragment_C[3]);
    packed_float4_t res;
    res.data[0] = fragment_D[0];
    res.data[1] = fragment_D[1];
    res.data[2] = fragment_D[2];
    res.data[3] = fragment_D[3];
    return res;
}
  
// m16n16k64 MMA
__device__ __forceinline__ static void mma_fp4_m16n8k32X4(packed_f32psum_t& out,
                                                          const packed_act_t* act,
                                                          const packed_wgt_t* wgt,
                                                          const packed_f32psum_t& psum,
                                                          uint32_t amscale,
                                                          uint32_t wmscale,
                                                          int ida,
                                                          int idb) {
    const int laneid = threadIdx.x % m16n16k64::WARP_SIZE;
    // A = [A1, A2],
    // B = [[A1, A3],
    //      [A2, A4]]
    // A1 * B1
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e2m1.e2m1.f32"
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};\n"
        : "=f"(out.data[0]), "=f"(out.data[1]), "=f"(out.data[2]), "=f"(out.data[3])
        : "r"(act[0].x), "r"(act[0].y), "r"(act[0].z), "r"(act[0].w),
          "r"(wgt[0].x), "r"(wgt[0].y),
          "f"(psum.data[0]), "f"(psum.data[1]), "f"(psum.data[2]), "f"(psum.data[3])
    );
    // A2 * B2 
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e2m1.e2m1.f32"
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};\n"
        : "=f"(out.data[0]), "=f"(out.data[1]), "=f"(out.data[2]), "=f"(out.data[3])
        : "r"(act[1].x), "r"(act[1].y), "r"(act[1].z), "r"(act[1].w),
          "r"(wgt[0].z), "r"(wgt[0].w),
          "f"(out.data[0]), "f"(out.data[1]), "f"(out.data[2]), "f"(out.data[3])
    );
    // 写回结果，给这个结果左乘右乘两个量化因子方阵
    __shared__ float out_sm[16 *8];
    
    const int index_out0 = laneid / 4 * 8 + laneid % 4 * 2;
    out_sm[index_out0] = out.data[0];
    out_sm[index_out0 + 1] = out.data[1];
    out_sm[index_out0 + 8 * 8] = out.data[2];
    out_sm[index_out0 + 8 * 8 + 1] = out.data[3];
    __syncthreads();
    //从量化因子构造左乘和右乘对角矩阵
    __shared__ fp8 left_scale[16*16];
    __shared__ fp8 right_scale[8*8];
    if(laneid==0){
        for(int i=0;i<16;i++){
            for(int j=0;j<16;j++){
                if(i == j)
                left_scale[i*16+j] = static_cast<fp8>(1.0f);
                else
                left_scale[i*16+j] = static_cast<fp8>(0.0f);
            }
        }
        for(int i=0;i<8;i++){
            for(int j=0;j<8;j++){
                if(i == j)
                right_scale[i*8+j] = static_cast<fp8>(1.0f);
                else
                right_scale[i*8+j] = static_cast<fp8>(0.0f);
            }
        }
    }
    __syncthreads();
    
    packed_float4_t temp_result;

    temp_result = ptx_m16n8k16_m16n8k8(left_scale,out_sm,right_scale);
    if(laneid ==4) {
        printf("temp_result[%d] = %8.3f\n",0,temp_result.data[0]);
        printf("temp_result[%d] = %8.3f\n",1,temp_result.data[1]);
        printf("temp_result[%d] = %8.3f\n",2,temp_result.data[2]);
        printf("temp_result[%d] = %8.3f\n",3,temp_result.data[3]);
    }
    out.data[0] = temp_result.data[0];
    out.data[1] = temp_result.data[1];
    out.data[2] = temp_result.data[2];
    out.data[3] = temp_result.data[3];

    out_sm[index_out0] = out.data[0];
    out_sm[index_out0 + 1] = out.data[1];
    out_sm[index_out0 + 8 * 8] = out.data[2];
    out_sm[index_out0 + 8 * 8 + 1] = out.data[3];
    __syncthreads();
    // if(laneid ==0) dumpEx1<float>(out_sm,16,8,"out_sm_left_dequantize");
    // A1 * B3
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e2m1.e2m1.f32"
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};\n"
        : "=f"(out.data[4]), "=f"(out.data[5]), "=f"(out.data[6]), "=f"(out.data[7])
        : "r"(act[0].x), "r"(act[0].y), "r"(act[0].z), "r"(act[0].w),
          "r"(wgt[1].x), "r"(wgt[1].y),
          "f"(psum.data[4]), "f"(psum.data[5]), "f"(psum.data[6]), "f"(psum.data[7])
    );
    // A2 * B4
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e2m1.e2m1.f32"
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};\n"
        : "=f"(out.data[4]), "=f"(out.data[5]), "=f"(out.data[6]), "=f"(out.data[7])
        : "r"(act[1].x), "r"(act[1].y), "r"(act[1].z), "r"(act[1].w),
          "r"(wgt[1].z), "r"(wgt[1].w),
          "f"(out.data[4]), "f"(out.data[5]), "f"(out.data[6]), "f"(out.data[7])
    );

    out_sm[index_out0] = out.data[4];
    out_sm[index_out0 + 1] = out.data[5];
    out_sm[index_out0 + 8 * 8] = out.data[6];
    out_sm[index_out0 + 8 * 8 + 1] = out.data[7];
    __syncthreads();
    // if(laneid ==0) dumpEx1<float>(out_sm,16,8,"out_sm_right");
    // TODO:构造反量化对角矩阵
    temp_result = ptx_m16n8k16_m16n8k8(left_scale,out_sm,right_scale);
    if(laneid ==0) {
        printf("temp_result[%d] = %8.3f\n",0,temp_result.data[0]);
        printf("temp_result[%d] = %8.3f\n",1,temp_result.data[1]);
        printf("temp_result[%d] = %8.3f\n",2,temp_result.data[2]);
        printf("temp_result[%d] = %8.3f\n",3,temp_result.data[3]);
    }
    out.data[4] = temp_result.data[0];
    out.data[5] = temp_result.data[1];
    out.data[6] = temp_result.data[2];
    out.data[7] = temp_result.data[3];
}

__global__ void mma_fp4_m16n16k64(float* D, e2m1* input_A, e2m1* input_B) {
    const int laneid = threadIdx.x % m16n16k64::WARP_SIZE;
    
    __shared__ uint8_t act[m16n16k64::M * m16n16k64::K];
    __shared__ uint8_t wgt[m16n16k64::N * m16n16k64::K];

    //先加载到share memory里
    const int size_a = m16n16k64::M * m16n16k64::K / m16n16k64::WARP_SIZE;
#pragma unroll
    for (int i = 0; i < size_a; i++) {
        act[laneid * size_a + i] = input_A[laneid * size_a + i].__x  << 2;      
    }

    const int size_b = m16n16k64::K * m16n16k64::N / m16n16k64::WARP_SIZE;
#pragma unroll
    for (int i = 0; i < size_b; i++) {
        int index = laneid * size_b + i;
        int row = index / m16n16k64::N;
        int col = index % m16n16k64::N;
        // 转置写入
        wgt[col * m16n16k64::K + row] = input_B[row * m16n16k64::N + col].__x << 2;
    }
    __syncthreads();

    packed_f32psum_t psum {0};
    packed_f32psum_t out {0};
    
    int index_a0 = (laneid / 4) * m16n16k64::K + (laneid % 4) * 4;
    int index_a1 = index_a0 + 32;
    packed_act_t packA[2]{
    {
            *(uint32_t*)&act[index_a0], 
            *(uint32_t*)&act[index_a0 + m16n16k64::K * 8],
            *(uint32_t*)&act[index_a0 + 16], 
            *(uint32_t*)&act[index_a0 + 16 + m16n16k64::K * 8]
        },
        {
            *(uint32_t*)&act[index_a1], 
            *(uint32_t*)&act[index_a1 + m16n16k64::K * 8],
            *(uint32_t*)&act[index_a1 + 16], 
            *(uint32_t*)&act[index_a1 + 16 + m16n16k64::K * 8]
        },
    };
    int index_b0 = (laneid / 4) * m16n16k64::K + (laneid % 4) * 4;
    int index_b1 = index_b0 + m16n16k64::K * 8;
    packed_act_t packB[2]{
        {
            *(uint32_t*)&wgt[index_b0], 
            *(uint32_t*)&wgt[index_b0 + 16], 
            *(uint32_t*)&wgt[index_b0 + 32], 
            *(uint32_t*)&wgt[index_b0 + 16 + 32]
        },
        {
            *(uint32_t*)&wgt[index_b1], 
            *(uint32_t*)&wgt[index_b1 + 16], 
            *(uint32_t*)&wgt[index_b1 + 32], 
            *(uint32_t*)&wgt[index_b1 + 16 + 32]
        }
    };

    mma_fp4_m16n8k32X4(out, packA, packB, psum,0,0,0,0);

    int index_d0 = (laneid / 4) * m16n16k64::N + (laneid % 4) * 2;
    D[index_d0] = out.data[0];
    D[index_d0 + 1] = out.data[1];
    D[index_d0 + m16n16k64::N * 8] = out.data[2];
    D[index_d0 + m16n16k64::N * 8 + 1] = out.data[3];

    int index_d1 = index_d0 + 8;
    D[index_d1] = out.data[4];
    D[index_d1 + 1] = out.data[5];
    D[index_d1 + m16n16k64::N * 8] = out.data[6];
    D[index_d1 + m16n16k64::N * 8 + 1] = out.data[7];

}

int main() {

    const char* envValue = std::getenv("MY_APP_MODE");
    printf("os:%s\n",envValue);
    int  flag = (envValue != nullptr) ? std::atoi(envValue) : 0;
    printf("flag = %d\n" , flag);

    switch (flag) {
        case 0:{
            printf("case 0 : flag = 0\n");
            break;
            
        }
        case 100:{
            printf("case 100 : flag = 100\n");
            break;
        }
        default:{
            printf("error case!!\n");
            break;
        }
    }

    e2m1 *host_a = new e2m1[m16n16k64::M * m16n16k64::K];
    e2m1 *host_b = new e2m1[m16n16k64::K * m16n16k64::N];
    float *host_d = new float[m16n16k64::M * m16n16k64::N];
    
    e2m1 *dev_a;
    e2m1 *dev_b;
    e2m1 *dev_bt;
    float *dev_d;
    
    CHECK_CUDA(cudaMalloc(&dev_a, sizeof(e2m1) * m16n16k64::M * m16n16k64::K));
    CHECK_CUDA(cudaMalloc(&dev_b, sizeof(e2m1) * m16n16k64::K * m16n16k64::N));
    CHECK_CUDA(cudaMalloc(&dev_bt, sizeof(e2m1)* m16n16k64::N * m16n16k64::K));
    CHECK_CUDA(cudaMalloc(&dev_d, sizeof(float)* m16n16k64::M * m16n16k64::N));
    
    e2m1 datas[] {
        e2m1(6.0), e2m1(-4.0), e2m1(-3.0), e2m1(-2.0),
        e2m1(-1.5), e2m1(-1.0), e2m1(-0.5), e2m1(-0.0), 
        e2m1(+0.0), e2m1(+0.5), e2m1(+1.0), e2m1(+1.5), 
        e2m1(+2.0), e2m1(+3.0), e2m1(+4.0), e2m1(+6.0),
    };

    for(int i = 0; i < m16n16k64::M * m16n16k64::K; ++i) host_a[i] = datas[i % 16];
    for(int i = 0; i < m16n16k64::K * m16n16k64::N; ++i) host_b[i] = datas[i % 16];
    for(int i = 0; i < m16n16k64::M * m16n16k64::N; ++i) host_d[i] = 0;

    CHECK_CUDA(cudaMemcpy(dev_a, host_a, sizeof(e2m1) *m16n16k64:: M * m16n16k64::K,cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dev_b, host_b, sizeof(e2m1) * m16n16k64::K * m16n16k64::N,cudaMemcpyHostToDevice));

    mma_fp4_m16n16k64<<<1, 32>>>(dev_d, dev_a, dev_b);

    cudaMemcpy(host_d, dev_d, sizeof(float) * m16n16k64::M * m16n16k64::N, cudaMemcpyDeviceToHost);
    for (int i = 0; i < m16n16k64::M; i++) {
        for (int j = 0; j < m16n16k64::N; j++) {
            printf("%10.2f ", host_d[i * m16n16k64::N + j]);
        }
        printf("\n");
    }
    cudaDeviceSynchronize();
    return 0;
}