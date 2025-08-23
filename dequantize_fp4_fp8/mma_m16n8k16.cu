#include <iostream>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cuda.h>
#include <mma.h>
#include <cuda_fp8.h>

using namespace nvcuda;
using fp8 = __nv_fp8_e4m3;

constexpr int WARP_SIZE = 32;
constexpr int M = 16;
constexpr int N = 8;    
constexpr int K = 16;


#define CHECK_CUDA(status)                                              \
  {                                                                     \
    cudaError_t error = status;                                         \
    if (error != cudaSuccess) {                                         \
      std::cerr << "Got bad cuda status: " << cudaGetErrorString(error) \
                << " at line: " << __LINE__ << std::endl;               \
      exit(EXIT_FAILURE);                                               \
    }                                                                   \
  }

//mma指令
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


// __device__ uint32_t pack_fp8(const fp8* data) {
//     uint32_t packed = 0;
//     for (int i = 0; i < 4; i++) {
//         packed |= (static_cast<uint32_t>(data[i].__x) << (i * 8));
//     }
//     return packed;
// }

__global__ void ptx_m16n8k16_kernel(fp8* input_A, fp8* input_B, half* input_C) {
    
    const size_t laneid = threadIdx.x % WARP_SIZE;
    
    __shared__ fp8 A[M * K];
    __shared__ fp8 B[K * N];  

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
        B[col * K + row] = input_B[row * N + col];
    }

    __syncthreads();
    
    clock_t begin=clock64();
    uint32_t RA[2]{0};
    uint32_t RB[1]{0};
    uint32_t RC[2]{0};
    
    int index_a = laneid / 4 * 16 + laneid % 4 * 4;
    RA[0] = *(uint32_t*)&A[index_a];
    RA[1] = *(uint32_t*)&A[index_a + K * 8];
    RB[0] = *(uint32_t*)&B[laneid * 4];

    //执行mma执行
    HMMA16816(RC[0], RC[1],
              RA[0], RA[1],
              RB[0],
              RC[0], RC[1]);   
    
    int index_c = laneid / 4 * 8 + laneid % 4 * 2;
    *(uint32_t*)&input_C[index_c] = RC[0];
    *(uint32_t*)&input_C[index_c + N * 8] = RC[1]; 

    clock_t end=clock64();
    
    if(laneid==0)
    {
        printf("ptx_mma_shared kernel e2e(cycles):%ld\n",end-begin);
    }    
}

void dump(half *host_c)
{
    for(int r=0;r<M;r++)
    {
       for(int c=0;c<N;c++)
       {
        printf("%8.3f ",__half2float(host_c[r*N+c]));
       }
       printf("\n");
    }
}

int main() {
    fp8 *host_a = new fp8[M*K];
    fp8 *host_b = new fp8[K*N];
    half *host_c = new half[M*N];
    
    fp8 *dev_a;
    fp8 *dev_b;
    half *dev_c;
    
    CHECK_CUDA(cudaMalloc(&dev_a, sizeof(fp8)*M*K));
    CHECK_CUDA(cudaMalloc(&dev_b, sizeof(fp8)*K*N));
    CHECK_CUDA(cudaMalloc(&dev_c, sizeof(half)*M*N));
    
    for(int i = 0; i < M*K; ++i) host_a[i] = static_cast<fp8>(i*0.01);
    for(int i = 0; i < K*N; ++i) host_b[i] = static_cast<fp8>(i*0.01);
    for(int i = 0; i < M*N; ++i) host_c[i] = 0;

    for(int j=0;j<1;j++)
    {
        CHECK_CUDA(cudaMemcpy(dev_a, host_a, sizeof(fp8)*M*K,cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(dev_b, host_b, sizeof(fp8)*K*N,cudaMemcpyHostToDevice));
      
        ptx_m16n8k16_kernel<<<1, 32>>>(dev_a, dev_b,dev_c);
        cudaDeviceSynchronize();
        
        cudaMemcpy(host_c, dev_c, sizeof(half)*M*N, cudaMemcpyDeviceToHost);
        dump(host_c);
    }
    
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    free(host_a);
    free(host_b);
    free(host_c);

    return 0;
}
