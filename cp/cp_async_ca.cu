#include <iostream>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cuda.h>
#include <mma.h>
#include <cuda_fp16.h>

#define WARP_SIZE 32

template<typename T>
__host__ __device__ 
void dump(T* matrix, const int M, const int N) {
    for(int i = 0; i < M; i++) {
        for(int j = 0; j < N; j++) {
            printf("%8.3f ", static_cast<float>(matrix[i * M + j]));
        }
        printf("\n"); 
    }
}

#define CHECK_CUDA(status)                                                \
    {                                                                      \
        cudaError_t error = status;                                        \
        if (error != cudaSuccess) {                                        \
            std::cerr << "Got bad cuda status: " << cudaGetErrorString(error) \
                      << " at line: " << __LINE__ << std::endl;            \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    }

// 异步加载数据
#define CP_ASYNC_CG(dst, src, Bytes) \
    asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(Bytes))

#define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;\n" ::)
#define CP_ASYNC_WAIT_GROUP(N) asm volatile("cp.async.wait_group %0;\n" ::"n"(N))

__global__ void ptx_cp_async_kernel(float* input_A, half* input_B, float* input_C, half* input_D) {
    const size_t laneid = threadIdx.x % WARP_SIZE;
    
    __shared__ float A[16 * 16];
    __shared__ half B[16 * 16]; 

    uint32_t a_smem_lane_addr_up = __cvta_generic_to_shared(&A[laneid * 4]); 
    uint32_t a_smem_lane_addr_down = __cvta_generic_to_shared(&A[laneid * 4 + 16 * 8]); 
    CP_ASYNC_CG(a_smem_lane_addr_up, &input_A[laneid * 4], 16);
    CP_ASYNC_CG(a_smem_lane_addr_down, &input_A[laneid * 4 + 16 * 8], 16);
    
    // uint32_t a_smem_lane_addr_up = __cvta_generic_to_shared(&A[laneid]); 
    // CP_ASYNC_CG(a_smem_lane_addr_up, &input_A[laneid], sizeof(float));


    CP_ASYNC_COMMIT_GROUP();
    CP_ASYNC_WAIT_GROUP(0);
    __syncthreads();
    
    if(laneid == 0) dump<float>(A, 16, 16); 
}

int main() {
    // hello<<<1,10>>>();
    float* host_a = new float[16 * 16];
    half* host_b = new half[16 * 16];
    float* host_c = new float[16 * 16];
    half* host_d = new half[16 * 16];

    float* dev_a;
    half* dev_b;
    float* dev_c;
    half* dev_d;

    CHECK_CUDA(cudaMalloc(&dev_a, sizeof(float) * 16 * 16));
    CHECK_CUDA(cudaMalloc(&dev_b, sizeof(half) * 16 * 16));
    CHECK_CUDA(cudaMalloc(&dev_c, sizeof(float) * 16 * 16));
    CHECK_CUDA(cudaMalloc(&dev_d, sizeof(half) * 16 * 16));

    for(int i = 0; i < 16; i++) {
        for(int j = 0; j < 16; j++) {
            host_a[i * 16 + j] = i * 16 + j;
            if(i == j) {
                host_b[i * 16 + j] = half(1.0f);
            } else {
                host_b[i * 16 + j] = half(0.0f);
            }
        }
    }
    printf("host_a:\n");
    dump<float>(host_a, 16, 16);
    printf("---------------------------------------------------\n");
    // printf("host_b:\n");
    // dump<half>(host_b, 16, 16);

    CHECK_CUDA(cudaMemcpy(dev_a, host_a, sizeof(float) * 16 * 16, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dev_b, host_b, sizeof(half) * 16 * 16, cudaMemcpyHostToDevice));

    ptx_cp_async_kernel<<<1, 32>>>(dev_a, dev_b, dev_c, dev_d);
    cudaDeviceSynchronize();
    
    cudaDeviceSynchronize();

    // 释放内存
    delete[] host_a;
    delete[] host_b;
    delete[] host_c;
    delete[] host_d;
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    cudaFree(dev_d);

    return 0;
}