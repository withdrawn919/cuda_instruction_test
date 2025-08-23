#include <iostream>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cuda.h>
#include <mma.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>

using namespace nvcuda;
using fp8 = __nv_fp8_e4m3;
using fp16 = __half;
using fp4 = __nv_fp4_e2m1;

constexpr int WARP_SIZE = 32;
constexpr int M = 16;
constexpr int N = 8;    
constexpr int K = 16;

constexpr int M1 = 16;
constexpr int N1 = 8;
constexpr int K1 = 8;

#define CHECK_CUDA(status)                                              \
  {                                                                     \
    cudaError_t error = status;                                         \
    if (error != cudaSuccess) {                                         \
      std::cerr << "Got bad cuda status: " << cudaGetErrorString(error) \
                << " at line: " << __LINE__ << std::endl;               \
      exit(EXIT_FAILURE);                                               \
    }                                                                   \
  }


// ================== 第一核函数相关实现 ==================
//mma指令,m16n8k16,输出为half类型,输入为fp8类型
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

__device__ uint32_t pack_fp8(const fp8 data0, const fp8 data1, const fp8 data2, const fp8 data3) {
    uint8_t uint0 = 0;
    uint8_t uint1 = 0;
    uint8_t uint2 = 0;
    uint8_t uint3 = 0;

    memcpy(&uint0, &data0, sizeof(fp8));
    memcpy(&uint1, &data1, sizeof(fp8));
    memcpy(&uint2, &data2, sizeof(fp8));
    memcpy(&uint3, &data3, sizeof(fp8));

    uint32_t packed = 0;
    packed |=uint3 << 24;
    packed |=uint2 << 16;
    packed |=uint1 << 8;
    packed |=uint0;
    
    return packed;
}

// ================== 第二核函数相关实现 ==================
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
  load_A_float_G(const float *A ,ARegisters &a_regs)
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
  load_B_fp8_G(const fp8 *B,BRegisters& b_regs)
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
    store_D(float *D, const DRegisters &d_regs, const int ldd = N1){
        const int laneid = threadIdx.x % 32;
        const int groupID = laneid >> 2; // laneid // 4
        const int threadID_in_group = laneid % 4; // laneid % 4
        D[groupID * ldd + threadID_in_group * 2 + (0 & 0x1)] = d_regs[0];
        D[groupID * ldd + threadID_in_group * 2 + (1 & 0x1)] = d_regs[1];
        D[(groupID + 8) * ldd + threadID_in_group * 2 + (4 & 0x1)] = d_regs[2];
        D[(groupID + 8) * ldd + threadID_in_group * 2 + (3 & 0x1)] = d_regs[3]; 
        // printf("D[%d][%d] = %f\n",0, laneid,d_regs[0]);
        // printf("D[%d][%d] = %f\n",1, laneid, d_regs[1]);
        // printf("D[%d][%d] = %f\n",2, laneid, d_regs[2]);
        // printf("D[%d][%d] = %f\n",3, laneid, d_regs[3]);  
    }
};
