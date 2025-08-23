#include <cuda_runtime.h>
#include <cuda.h>
#include <mma.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>
#include <cstdio>

#include "wmma.cuh"
#include "wmma_utils.cuh"
#include "../config.h"


using packed_act_t = uint4;
using packed_wgt_t = uint4;
using fp4 = __nv_fp4_e2m1;
using fp8 = __nv_fp8_e4m3;

// 16 * 16 matrix
struct alignas(32) packed_f32psum_t {
    float data[4];
};

#define USE_SHARED_MEMORY 1

constexpr int AM = 16;
constexpr int AN = 16;  
constexpr int AK = 64;

// D=A×B×C
__device__ void ptx_m16n8k16_m16n8k8_kernel(fp8* input_A, float* input_B, fp8* input_C, float* output_D, const int ldb) {
    //---------------- 第一个tensor core计算开始------------
    const size_t laneid = threadIdx.x % WARP_SIZE;
    __shared__ fp8 A[M * K];
    __shared__ fp8 B[K * N];  
    if(USE_SHARED_MEMORY){

    //先加载到share memory里
    const int size_a = M * K / WARP_SIZE;
    const int size_b = K * N / WARP_SIZE;
#pragma unroll
    for (int i = 0; i < size_a; i++) {
        A[laneid * size_a + i] = input_A[laneid * size_a + i];     
    }
#pragma unroll
    for (int i = 0; i < size_b; i++) {
        int flat = laneid * size_b + i;
        int row = flat / N;
        int col = flat % N;
        // 转置写入
        B[col * K + row] = static_cast<fp8>(input_B[row * ldb + col]);
    }
    __syncthreads();
    }

    uint32_t RA[2]{0};
    uint32_t RB[1]{0};
    uint32_t RC[2]{0};
    uint32_t RD[2]{0};
    
    int index_a = laneid / 4 * 16 + laneid % 4 * 4;
    int index_b = laneid % 4 * N * 4 + laneid /4;
    if(USE_SHARED_MEMORY){
        RA[0] = *(uint32_t*)&A[index_a];
        RA[1] = *(uint32_t*)&A[index_a + K * 8];
        RB[0] = *(uint32_t*)&B[laneid * 4];
    } else {
        RA[0] = *(uint32_t*)&input_A[index_a];
        RA[1] = *(uint32_t*)&input_A[index_a + K * 8];
        RB[0] = pack_fp8(B[index_b],B[index_b + N], B[index_b + N * 2], B[index_b + N * 3]);
    }

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
    SM75_16x8x8_F32F16F16F32_TN::BRegisters fragment_B;
    SM75_16x8x8_F32F16F16F32_TN::load_B_fp8_G(input_C, fragment_B);
    // initialize C
    SM75_16x8x8_F32F16F16F32_TN::CRegisters fragment_C = {0.0f, 0.0f, 0.0f, 0.0f};
    // initialize D
    SM75_16x8x8_F32F16F16F32_TN::DRegisters fragment_D = {0.0f, 0.0f, 0.0f, 0.0f};
    // execute the MMA operation
    SM75_16x8x8_F32F16F16F32_TN::fma(fragment_D[0], fragment_D[1], fragment_D[2], fragment_D[3],
        fragment_A[0], fragment_A[1],
        fragment_B[0],
        fragment_C[0], fragment_C[1], fragment_C[2], fragment_C[3]);
    SM75_16x8x8_F32F16F16F32_TN::store_D(output_D, fragment_D, ldb);
}


// m16n16k64 MMA
__device__ __forceinline__ static void mma_fp4_m16n8k64(packed_f32psum_t& out,
                                                        const packed_act_t& act,
                                                        const packed_wgt_t& wgt,
                                                        const packed_f32psum_t& psum) {
    // D1 = A1 * B1
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e2m1.e2m1.f32"
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};\n"
        : "=f"(out.data[0]), "=f"(out.data[1]), "=f"(out.data[2]), "=f"(out.data[3])
        : "r"(act.x), "r"(act.y), "r"(act.z), "r"(act.w),
          "r"(wgt.x), "r"(wgt.y),
          "f"(psum.data[0]), "f"(psum.data[1]), "f"(psum.data[2]), "f"(psum.data[3])
    );
    // D = A2 * B2 + D1
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e2m1.e2m1.f32"
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};\n"
        : "=f"(out.data[0]), "=f"(out.data[1]), "=f"(out.data[2]), "=f"(out.data[3])
        : "r"(act.x), "r"(act.y), "r"(act.z), "r"(act.w),
          "r"(wgt.z), "r"(wgt.w),
          "f"(out.data[0]), "f"(out.data[1]), "f"(out.data[2]), "f"(out.data[3])
    );
}

__global__ void mma_fp4_m16n16k64(float* D, 
                                  fp4* input_A, 
                                  fp4* input_B, 
                                  fp8* amscale0,
                                  fp8* amscale1,
                                  fp8* wmscale0,
                                  fp8* wmscale1) {
    const int laneid = threadIdx.x % WARP_SIZE;
    
    __shared__ uint8_t act[AM * AK];
    __shared__ uint8_t wgt[AN * AK];

    //先加载到share memory里
    const int size_a = AM * AK / WARP_SIZE;
#pragma unroll
    for (int i = 0; i < size_a; i++) {
        act[laneid * size_a + i] = input_A[laneid * size_a + i].__x  << 2;      
    }

    const int size_b = AK * AN / WARP_SIZE;
#pragma unroll
    for (int i = 0; i < size_b; i++) {
        int index = laneid * size_b + i;
        int row = index / AN;
        int col = index % AN;
        // 转置写入
        wgt[col * AK + row] = input_B[row * AN + col].__x << 2;
    }
    __syncthreads();
    
    int index_a0 = (laneid / 4) * AK + (laneid % 4) * 4;
    int index_a1 = index_a0 + 32;
    packed_act_t packA[2]{
    {
            *(uint32_t*)&act[index_a0], 
            *(uint32_t*)&act[index_a0 + AK * 8],
            *(uint32_t*)&act[index_a0 + 16], 
            *(uint32_t*)&act[index_a0 + 16 + AK * 8]
        },
        {
            *(uint32_t*)&act[index_a1], 
            *(uint32_t*)&act[index_a1 + AK * 8],
            *(uint32_t*)&act[index_a1 + 16], 
            *(uint32_t*)&act[index_a1 + 16 + AK * 8]
        },
    };
    int index_b0 = (laneid / 4) * AK + (laneid % 4) * 4;
    int index_b1 = index_b0 + AK * 8;
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

    packed_f32psum_t psum[2] {0};
    packed_f32psum_t out[2] {0};
    // D1 = A1B1 + A2B2
    mma_fp4_m16n8k64(out[0], packA[0], packB[0], psum[0]);
    // 计算完之后写回
    int index_d0 = (laneid / 4) * AN + (laneid % 4) * 2;
    D[index_d0] = out[0].data[0];
    D[index_d0 + 1] = out[0].data[1];
    D[index_d0 + AN * 8] = out[0].data[2];
    D[index_d0 + AN * 8 + 1] = out[0].data[3];
    // 反量化
    ptx_m16n8k16_m16n8k8_kernel(amscale0, D, wmscale0, D, AN);

    // D2 = A1B3 + A2B4
    mma_fp4_m16n8k64(out[1], packA[1], packB[1], psum[1]);
    int index_d1 = index_d0 + 8;
    D[index_d1] = out[1].data[0];
    D[index_d1 + 1] = out[1].data[1];
    D[index_d1 + AN * 8] = out[1].data[2];
    D[index_d1 + AN * 8 + 1] = out[1].data[3];
    ptx_m16n8k16_m16n8k8_kernel(amscale1, D + 8, wmscale1, D  + 8, AN);
}

int main() {
    fp4 *host_a = new fp4[AM * AK];
    fp4 *host_b = new fp4[AK * AN];
    float *host_d = new float[AM * AN];
    
    fp4 *dev_a;
    fp4 *dev_b;
    float *dev_d;

    CHECK_CUDA(cudaMalloc(&dev_a, sizeof(fp4) * AM * AK));
    CHECK_CUDA(cudaMalloc(&dev_b, sizeof(fp4) * AK * AN));
    CHECK_CUDA(cudaMalloc(&dev_d, sizeof(float) * AM * AN));

    fp8* host_amscale0 = new fp8[AM * AM];
    fp8* host_amscale1 = new fp8[AM * AM];
    fp8* host_wmscale0 = new fp8[AN / 2 * AN / 2];
    fp8* host_wmscale1 = new fp8[AN / 2 * AN / 2];

    fp8* dev_amscale0;
    fp8* dev_amscale1;
    fp8* dev_wmscale0;
    fp8* dev_wmscale1;
    
    CHECK_CUDA(cudaMalloc(&dev_amscale0, sizeof(fp8) * AM * AM));
    CHECK_CUDA(cudaMalloc(&dev_amscale1, sizeof(fp8) * AM * AM));
    CHECK_CUDA(cudaMalloc(&dev_wmscale0, sizeof(fp8) * AN / 2 * AN / 2));
    CHECK_CUDA(cudaMalloc(&dev_wmscale1, sizeof(fp8) * AN / 2 * AN / 2));
    
    fp4 datas[] {
        fp4(6.0), fp4(-4.0), fp4(-3.0), fp4(-2.0),
        fp4(-1.5), fp4(-1.0), fp4(-0.5), fp4(-0.0), 
        fp4(+0.0), fp4(+0.5), fp4(+1.0), fp4(+1.5), 
        fp4(+2.0), fp4(+3.0), fp4(+4.0), fp4(+6.0),
    };

    for(int i = 0; i < AM * AK; ++i) host_a[i] = datas[i % 16];
    for(int i = 0; i < AK * AN; ++i) host_b[i] = datas[i % 16];
    for(int i = 0; i < AM * AN; ++i) host_d[i] = 0;

    CHECK_CUDA(cudaMemcpy(dev_a, host_a, sizeof(fp4) * AM * AK,cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dev_b, host_b, sizeof(fp4) * AK * AN,cudaMemcpyHostToDevice));

    for(int i = 0; i < AM; i++) {
        for(int j = 0; j < AM; j++) {
            host_amscale0[i * AM + j] = fp8(i == j ? 0.5: 0);
            host_amscale1[i * AM + j] = fp8(i == j ? 0.5: 0);
        }
    }
    for(int i = 0; i < AN / 2; i++) {
        for(int j = 0; j < AN / 2; j++) {
            host_wmscale0[i * AN / 2 + j] = fp8(i == j ? -1: 0);
            host_wmscale1[i * AN / 2 + j] = fp8(i == j ? -1: 0);
        }
    }

    CHECK_CUDA(cudaMemcpy(dev_amscale0, host_amscale0, sizeof(fp8) * AM * AM,cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dev_amscale1, host_amscale1, sizeof(fp8) * AM * AM,cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dev_wmscale0, host_wmscale0, sizeof(fp8) * AN / 2 * AN / 2,cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dev_wmscale1, host_wmscale1, sizeof(fp8) * AN / 2 * AN / 2,cudaMemcpyHostToDevice));

    float time;
    int step = 10000;
    cudaEvent_t start,end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaEventRecord(start, 0);
    for (int i = 0; i < step; i++)
        mma_fp4_m16n16k64<<<1, 32>>>(dev_d, dev_a, dev_b, dev_amscale0, dev_amscale1, dev_wmscale0, dev_wmscale1);
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);

    cudaEventElapsedTime(&time, start, end);
    printf("time cost: %lfms\n", time / step);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaMemcpy(host_d, dev_d, sizeof(float) * AM * AN, cudaMemcpyDeviceToHost);
    for (int i = 0; i < AM; i++) {
        for (int j = 0; j < AN; j++) {
            printf("%6.2f ", host_d[i * AN + j]);
        }
        printf("\n");
    }
    
    return 0;
}