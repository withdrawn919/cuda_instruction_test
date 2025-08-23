#include <cuda_runtime.h>
#include <cuda.h>
#include <mma.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>
#include <cstdio>
#include <iostream>


using packed_act_t = uint4;
using packed_wgt_t = uint4;
using e4m3 = __nv_fp8_e4m3;
using e2m1 = __nv_fp4_e2m1;

// 16 * 16 matrix
struct alignas(32) packed_f32psum_t {
    float data[8];
};

constexpr int M = 16;
constexpr int N = 16;  
constexpr int K = 64;
constexpr int WARP_SIZE = 32;

#define CHECK_CUDA(status)                                              \
  {                                                                     \
    cudaError_t error = status;                                         \
    if (error != cudaSuccess) {                                         \
      std::cerr << "Got bad cuda status: " << cudaGetErrorString(error) \
                << " at line: " << __LINE__ << std::endl;               \
      exit(EXIT_FAILURE);                                               \
    }                                                                   \
  }

// m16n16k64 MMA
__device__ __forceinline__ static void mma_fp4_m16n8k32X4(packed_f32psum_t& out,
                                                          const packed_act_t* act,
                                                          const packed_wgt_t* wgt,
                                                          const packed_f32psum_t& psum) {
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
}

__global__ void mma_fp4_m16n16k64(float* D, e2m1* input_A, e2m1* input_B) {
    const int laneid = threadIdx.x % WARP_SIZE;
    
    __shared__ uint8_t act[M * K];
    __shared__ uint8_t wgt[N * K];

    //先加载到share memory里
    const int size_a = M * K / WARP_SIZE;
#pragma unroll
    for (int i = 0; i < size_a; i++) {
        act[laneid * size_a + i] = input_A[laneid * size_a + i].__x  << 2;      
    }

    const int size_b = K * N / WARP_SIZE;
#pragma unroll
    for (int i = 0; i < size_b; i++) {
        int index = laneid * size_b + i;
        int row = index / N;
        int col = index % N;
        // 转置写入
        wgt[col * K + row] = input_B[row * N + col].__x << 2;
    }
    __syncthreads();

    packed_f32psum_t psum {0};
    packed_f32psum_t out {0};
    
    int index_a0 = (laneid / 4) * K + (laneid % 4) * 4;
    int index_a1 = index_a0 + 32;
    packed_act_t packA[2]{
    {
            *(uint32_t*)&act[index_a0], 
            *(uint32_t*)&act[index_a0 + K * 8],
            *(uint32_t*)&act[index_a0 + 16], 
            *(uint32_t*)&act[index_a0 + 16 + K * 8]
        },
        {
            *(uint32_t*)&act[index_a1], 
            *(uint32_t*)&act[index_a1 + K * 8],
            *(uint32_t*)&act[index_a1 + 16], 
            *(uint32_t*)&act[index_a1 + 16 + K * 8]
        },
    };
    int index_b0 = (laneid / 4) * K + (laneid % 4) * 4;
    int index_b1 = index_b0 + K * 8;
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

    mma_fp4_m16n8k32X4(out, packA, packB, psum);

    int index_d0 = (laneid / 4) * N + (laneid % 4) * 2;
    D[index_d0] = out.data[0];
    D[index_d0 + 1] = out.data[1];
    D[index_d0 + N * 8] = out.data[2];
    D[index_d0 + N * 8 + 1] = out.data[3];

    int index_d1 = index_d0 + 8;
    D[index_d1] = out.data[4];
    D[index_d1 + 1] = out.data[5];
    D[index_d1 + N * 8] = out.data[6];
    D[index_d1 + N * 8 + 1] = out.data[7];
}

int main() {
    e2m1 *host_a = new e2m1[M * K];
    e2m1 *host_b = new e2m1[K * N];
    float *host_d = new float[M * N];
    
    e2m1 *dev_a;
    e2m1 *dev_b;
    e2m1 *dev_bt;
    float *dev_d;
    
    CHECK_CUDA(cudaMalloc(&dev_a, sizeof(e2m1) * M * K));
    CHECK_CUDA(cudaMalloc(&dev_b, sizeof(e2m1) * K * N));
    CHECK_CUDA(cudaMalloc(&dev_bt, sizeof(e2m1)* N * K));
    CHECK_CUDA(cudaMalloc(&dev_d, sizeof(float)* M * N));
    
    e2m1 datas[] {
        e2m1(6.0), e2m1(-4.0), e2m1(-3.0), e2m1(-2.0),
        e2m1(-1.5), e2m1(-1.0), e2m1(-0.5), e2m1(-0.0), 
        e2m1(+0.0), e2m1(+0.5), e2m1(+1.0), e2m1(+1.5), 
        e2m1(+2.0), e2m1(+3.0), e2m1(+4.0), e2m1(+6.0),
    };

    for(int i = 0; i < M * K; ++i) host_a[i] = datas[i % 16];
    for(int i = 0; i < K * N; ++i) host_b[i] = datas[i % 16];
    for(int i = 0; i < M * N; ++i) host_d[i] = 0;

    CHECK_CUDA(cudaMemcpy(dev_a, host_a, sizeof(e2m1) * M * K,cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dev_b, host_b, sizeof(e2m1) * K * N,cudaMemcpyHostToDevice));

    mma_fp4_m16n16k64<<<1, 32>>>(dev_d, dev_a, dev_b);

    cudaMemcpy(host_d, dev_d, sizeof(float) * M * N, cudaMemcpyDeviceToHost);
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%6.2f ", host_d[i * N + j]);
        }
        printf("\n");
    }
    
    return 0;
}