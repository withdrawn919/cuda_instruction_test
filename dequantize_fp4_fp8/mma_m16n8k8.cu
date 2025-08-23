
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <cstdio>
#include <stdio.h>
#include "../config.h"

constexpr int M1 = 16, N1 = 8, K1 = 8;
using fp8 = __nv_fp8_e4m3;
using fp16 = __half;

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
  load_A(const float *A ,ARegisters &a_regs)
  {
    const int laneid = threadIdx.x % 32;
    const int groupID = laneid >> 2; // laneid // 4
    const int threadID_in_group = laneid % 4; // laneid % 4

    float A0 = A[groupID * K1 + threadID_in_group * 2 + (0 & 0x1)];
    float A1 = A[groupID * K1 + threadID_in_group * 2 + (1 & 0x1)];
    float A2 = A[(groupID + 8) * K1 + threadID_in_group * 2 + (4 & 0x1)];
    float A3 = A[(groupID + 8) * K1 + threadID_in_group * 2 + (3 & 0x1)];


    __half2 half2_A0_A1 = __half2{__float2half(A0), __float2half(A1)};
    __half2 half2_A2_A3 = __half2{__float2half(A2), __float2half(A3)};
    // printf("A[%d][%d] = %f\n",0,laneid, static_cast<float>(half2_A0_A1.x));
    // printf("A[%d][%d] = %f\n",1,laneid, static_cast<float>(half2_A0_A1.y));
    // printf("A[%d][%d] = %f\n",2,laneid, static_cast<float>(half2_A2_A3.x));
    // printf("A[%d][%d] = %f\n",3,laneid, static_cast<float>(half2_A2_A3.y));

    a_regs[0] = *reinterpret_cast<uint32_t*>(&half2_A0_A1);
    a_regs[1] = *reinterpret_cast<uint32_t*>(&half2_A2_A3);

  }

  __device__ static void
  load_B(const fp8 *B,BRegisters& b_regs)
  {
    const int laneid = threadIdx.x % 32;
    const int groupID = laneid >> 2; // laneid // 4
    const int threadID_in_group = laneid % 4; // laneid % 4

    fp8 B0 = B[(threadID_in_group * 2 + 0) * K1 + groupID];
    fp8 B1 = B[(threadID_in_group * 2 + 1) * K1 + groupID];

    __half2 half2_B0_B1 = __half2{static_cast<half>(B0), static_cast<half>(B1)};
    // printf("B[%d][%d] = %f\n",laneid,1, static_cast<float>(half2_B0_B1.x));
    // printf("B[%d][%d] = %f\n",laneid,0, static_cast<float>(half2_B0_B1.y));
    b_regs[0] = *reinterpret_cast<uint32_t*>(&half2_B0_B1);
  }

  __device__ static void
    store_D(float *D, const DRegisters &d_regs){
        const int laneid = threadIdx.x % 32;
        const int groupID = laneid >> 2; // laneid // 4
        const int threadID_in_group = laneid % 4; // laneid % 4
        D[groupID * N1 + threadID_in_group * 2 + (0 & 0x1)] = d_regs[0];
        D[groupID * N1 + threadID_in_group * 2 + (1 & 0x1)] = d_regs[1];
        D[(groupID + 8) * N1 + threadID_in_group * 2 + (4 & 0x1)] = d_regs[2];
        D[(groupID + 8) * N1 + threadID_in_group * 2 + (3 & 0x1)] = d_regs[3]; 
        // printf("D[%d][%d] = %f\n",0, laneid,d_regs[0]);
        // printf("D[%d][%d] = %f\n",1, laneid, d_regs[1]);
        // printf("D[%d][%d] = %f\n",2, laneid, d_regs[2]);
        // printf("D[%d][%d] = %f\n",3, laneid, d_regs[3]);  
    }
};



__global__ void gemm_m16n8k8_fp32_fp8(float *D,
                                      const float *A,
                                      const fp8 *B,
                                      const float *C)
{
    SM75_16x8x8_F32F16F16F32_TN::ARegisters fragment_A;
    SM75_16x8x8_F32F16F16F32_TN::load_A(A,fragment_A);
    SM75_16x8x8_F32F16F16F32_TN::BRegisters fragment_B;
    SM75_16x8x8_F32F16F16F32_TN::load_B(B,fragment_B);
    SM75_16x8x8_F32F16F16F32_TN::CRegisters fragment_C = {0.0f, 0.0f, 0.0f, 0.0f};
    SM75_16x8x8_F32F16F16F32_TN::DRegisters fragment_D = {0.0f, 0.0f, 0.0f, 0.0f};
    SM75_16x8x8_F32F16F16F32_TN::fma(fragment_D[0], fragment_D[1], fragment_D[2], fragment_D[3],
                                      fragment_A[0], fragment_A[1],
                                      fragment_B[0],
                                      fragment_D[0], fragment_D[1], fragment_D[2], fragment_D[3]);
    SM75_16x8x8_F32F16F16F32_TN::store_D(D, fragment_D);
}
int main(){
    // 分配主机内存
    float *h_A = static_cast<float*>(malloc(M1 * K1 * sizeof(float)));
    fp8   *h_B = static_cast<fp8*>( malloc(K1 * N1 * sizeof(fp8)));
    float *h_D = static_cast<float*>(malloc(M1 * N1 * sizeof(float)));
    // 初始化数值 A 是i++,
    for (int i = 0; i < M1 * K1; ++i) {
        h_A[i] = static_cast<float>(i);
    }
    //  B是单位矩阵
   for(int i=0;i<K1;i++){
        for(int j=0;j<N1;j++){
            if(i == j)
                h_B[i*N1+j] = static_cast<fp8>(1.0f);
            else
                h_B[i*N1+j] = static_cast<fp8>(0.0f);
        }
    }
    // D矩阵初始化为0
    for (int i = 0; i < M1 * N1; ++i) {
        h_D[i] = 0.0f;
    }
    for(int i=0;i<M1;i++){
        for(int j=0;j<K1;j++){
            printf("%f ", h_A[i * K1 + j]);
        }
        printf("\n");
    }
    printf("-------------------------------------------\n");
    for(int i=0;i<K1;i++){
        for(int j=0;j<N1;j++){
            printf("%f ", (float)(h_B[i * K1 + j]));
        }
        printf("\n");
    }
    // 分配设备内存
    float *d_A;
    fp8   *d_B;
    float *d_D;
    cudaMalloc(&d_A, M1 * K1 * sizeof(float));
    cudaMalloc(&d_B, K1 * N1 * sizeof(fp8));
    cudaMalloc(&d_D, M1 * N1 * sizeof(float));

    // 拷贝数据到设备
    cudaMemcpy(d_A, h_A, M1 * K1 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K1 * N1 * sizeof(fp8), cudaMemcpyHostToDevice);

    // 启动核函数
    dim3 block(32);
    dim3 grid(1);
    #ifdef TEST_TIME
    // 创建CUDA事件计时
    // warm up
    for(int i = 0; i < ITERS ; i++) {
        // Warm up
        gemm_m16n8k8_fp32_fp8<<<grid, block>>>(d_D, d_A, d_B, d_D);    
    }
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for(int i = 0; i < ITERS ; i++) {
    #endif
    gemm_m16n8k8_fp32_fp8<<<grid, block>>>(d_D, d_A, d_B, d_D);
    #ifdef TEST_TIME
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    printf("单次反量化运行时间: %.12f ms\n", (float)(ms/ITERS));//
    #endif
    // 拷贝结果回主机
   
    cudaMemcpy(h_D, d_D, M1 * N1 * sizeof(float), cudaMemcpyDeviceToHost);

    // 打印结果
    for (int i = 0; i < M1; ++i) {
        for (int j = 0; j < N1; ++j) {
            printf("%f ", h_D[i * N1 + j]);
        }
        printf("\n");
    }

    // 清理资源
    free(h_A);
    free(h_B);
    free(h_D);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_D);

    return 0;
}