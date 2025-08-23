#include <cuda_runtime.h>
#include <cuda.h>
#include <mma.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>
#include <cstdio>
#include <iostream>


using packed_act_t = uint4;
using packed_wgt_t = uint4;
using fp4 = __nv_fp4_e2m1;
using fp8 = __nv_fp8_e4m3;


#define CHECK_CUDA(status)                                                  \
{                                                                           \
    cudaError_t error = status;                                             \
    if (error != cudaSuccess) {                                             \
        std::cerr << "Got bad cuda status: " << cudaGetErrorString(error)   \
                << " at line: " << __LINE__ << std::endl;                   \
        exit(EXIT_FAILURE);                                                 \
    }                                                                       \
}

// 16 * 16 matrix
struct alignas(32) packed_f32psum_t {
    float data[4];
};

constexpr int WARP_SIZE = 32;
constexpr int M = 16;
constexpr int N = 16;  
constexpr int K = 64;


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

    packed_f32psum_t psum[2] {0};
    packed_f32psum_t out[2] {0};
    // D1 = A1B1 + A2B2
    mma_fp4_m16n8k64(out[0], packA[0], packB[0], psum[0]);
    // 计算完之后写回
    int groupID           = laneid >> 2;
    int threadID_in_group = laneid % 4;
    int index_d0 = groupID * N + threadID_in_group * 2;
    D[index_d0] = out[0].data[0] * static_cast<float>(amscale0[groupID]) * static_cast<float>(wmscale0[threadID_in_group]);
    D[index_d0 + 1] = out[0].data[1] * static_cast<float>(amscale0[groupID]) * static_cast<float>(wmscale0[threadID_in_group + 1]);
    D[index_d0 + N * 8] = out[0].data[2] * static_cast<float>(amscale0[groupID + 1]) * static_cast<float>(wmscale0[threadID_in_group]);
    D[index_d0 + N * 8 + 1] = out[0].data[3] * static_cast<float>(amscale0[groupID + 1]) * static_cast<float>(wmscale0[threadID_in_group + 1]);

    // D2 = A1B3 + A2B4
    mma_fp4_m16n8k64(out[1], packA[1], packB[1], psum[1]);
    int index_d1 = index_d0 + 8;
    D[index_d1] = out[1].data[0] * static_cast<float>(amscale1[groupID]) * static_cast<float>(wmscale1[threadID_in_group]);
    D[index_d1 + 1] = out[1].data[1] * static_cast<float>(amscale1[groupID]) * static_cast<float>(wmscale1[threadID_in_group + 1]);
    D[index_d1 + N * 8] = out[1].data[2] * static_cast<float>(amscale1[groupID + 1]) * static_cast<float>(wmscale1[threadID_in_group]);
    D[index_d1 + N * 8 + 1] = out[1].data[3] * static_cast<float>(amscale1[groupID + 1]) * static_cast<float>(wmscale1[threadID_in_group + 1]);
}

int main() {
    fp4 *host_a = new fp4[M * K];
    fp4 *host_b = new fp4[K * N];
    float *host_d = new float[M * N];
    
    fp4 *dev_a;
    fp4 *dev_b;
    float *dev_d;

    CHECK_CUDA(cudaMalloc(&dev_a, sizeof(fp4) * M * K));
    CHECK_CUDA(cudaMalloc(&dev_b, sizeof(fp4) * K * N));
    CHECK_CUDA(cudaMalloc(&dev_d, sizeof(float) * M * N));

    fp8* host_amscale0 = new fp8[M];
    fp8* host_amscale1 = new fp8[M];
    fp8* host_wmscale0 = new fp8[N / 2];
    fp8* host_wmscale1 = new fp8[N / 2];

    fp8* dev_amscale0;
    fp8* dev_amscale1;
    fp8* dev_wmscale0;
    fp8* dev_wmscale1;
    
    CHECK_CUDA(cudaMalloc(&dev_amscale0, sizeof(fp8) * M));
    CHECK_CUDA(cudaMalloc(&dev_amscale1, sizeof(fp8) * M));
    CHECK_CUDA(cudaMalloc(&dev_wmscale0, sizeof(fp8) * N / 2));
    CHECK_CUDA(cudaMalloc(&dev_wmscale1, sizeof(fp8) * N / 2));
    
    fp4 datas[] {
        fp4(6.0), fp4(-4.0), fp4(-3.0), fp4(-2.0),
        fp4(-1.5), fp4(-1.0), fp4(-0.5), fp4(-0.0), 
        fp4(+0.0), fp4(+0.5), fp4(+1.0), fp4(+1.5), 
        fp4(+2.0), fp4(+3.0), fp4(+4.0), fp4(+6.0),
    };

    for(int i = 0; i < M * K; ++i) host_a[i] = datas[i % 16];
    for(int i = 0; i < K * N; ++i) host_b[i] = datas[i % 16];
    for(int i = 0; i < M * N; ++i) host_d[i] = 0;

    CHECK_CUDA(cudaMemcpy(dev_a, host_a, sizeof(fp4) * M * K,cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dev_b, host_b, sizeof(fp4) * K * N,cudaMemcpyHostToDevice));

    for(int i = 0; i < M; i++) {
        host_amscale0[i] = fp8(0.5);
        host_amscale1[i] = fp8(0.5);
    }
    for(int i = 0; i < N / 2; i++) {
        host_wmscale0[i] = fp8(-1);
        host_wmscale1[i] = fp8(-1);
    }

    CHECK_CUDA(cudaMemcpy(dev_amscale0, host_amscale0, sizeof(fp8) * M,cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dev_amscale1, host_amscale1, sizeof(fp8) * M,cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dev_wmscale0, host_wmscale0, sizeof(fp8) * N / 2,cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dev_wmscale1, host_wmscale1, sizeof(fp8) * N / 2,cudaMemcpyHostToDevice));

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
    
    cudaMemcpy(host_d, dev_d, sizeof(float) * M * N, cudaMemcpyDeviceToHost);
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%6.2f ", host_d[i * N + j]);
        }
        printf("\n");
    }
    
    return 0;
}