#include <iostream>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cuda.h>
#include <mma.h>

#define LDMATRIX_X1(R0, addr)                                             \
    asm volatile("ldmatrix.sync.aligned.x1.m8n8.shared.b16 {%0}, [%1];\n" \
                 : "=r"(R0)                                               \
                 : "l"(addr))

#define LDMATRIX_X2(R0, R1, addr)                                             \
    asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n" \
                 : "=r"(R0), "=r"(R1)                                         \
                 : "l"(addr))

#define LDMATRIX_X4(R0, R1, R2, R3, addr)                                             \
    asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n" \
                 : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3)                             \
                 : "l"(addr))





__global__ void ldmatrix_m8n8_x1(){
    __shared__ half A[8*8];
    const size_t laneid = threadIdx.x % 32;
    A[laneid]=__float2half(laneid * 1.0f);
    A[laneid + 32]=__float2half(laneid * 2.0f);
    __syncthreads();

    uint32_t RA;

    LDMATRIX_X1(RA, __cvta_generic_to_shared(&A[(laneid * 8) % (8 * 8)]));
}

__global__ void ldmatrix_m8n8_x2(){
    __shared__ half A[16*8];
    const size_t laneid = threadIdx.x % 32;
    A[laneid]=__float2half(laneid * 1.0f);
    A[laneid + 32]=__float2half(laneid * 2.0f);
    A[laneid + 32 + 32]=__float2half(laneid * 3.0f);
    A[laneid + 32 + 32 + 32]=__float2half(laneid * 4.0f);
    __syncthreads();

    uint32_t RA[2];

    LDMATRIX_X2(RA[0], RA[1], __cvta_generic_to_shared(&A[(laneid * 8) % (16 * 8)]));
}

__global__ void ldmatrix_m8n8_x4(){
    __shared__ half A[32*8];
    const size_t laneid = threadIdx.x % 32;
    A[laneid]=__float2half(laneid * 1.0f);
    A[laneid + 32]=__float2half(laneid * 2.0f);
    A[laneid + 32 + 32]=__float2half(laneid * 3.0f);
    A[laneid + 32 + 32 + 32]=__float2half(laneid * 4.0f);
    A[laneid + 32 + 32 + 32 + 32]=__float2half(laneid * 5.0f);
    A[laneid + 32 + 32 + 32 + 32 + 32]=__float2half(laneid * 6.0f);
    A[laneid + 32 + 32 + 32 + 32 + 32 + 32]=__float2half(laneid * 7.0f);
    A[laneid + 32 + 32 + 32 + 32 + 32 + 32 + 32]=__float2half(laneid * 8.0f);
    __syncthreads();

    uint32_t RA[4];

    LDMATRIX_X4(RA[0], RA[1], RA[2], RA[3], __cvta_generic_to_shared(&A[(laneid * 8) % (32 * 8)]));
}

int main(){
    ldmatrix_m8n8_x1<<<1,32>>>();
    cudaDeviceSynchronize();
    ldmatrix_m8n8_x2<<<1,32>>>();
    cudaDeviceSynchronize();
    ldmatrix_m8n8_x4<<<1,32>>>();
    cudaDeviceSynchronize();
    return 0;

}