#include <cuda_runtime.h>
#include <iostream>
#include <stdlib.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include "../config.h"


#define VEC 16
typedef __nv_fp8_e4m3 fp8;
typedef __nv_fp8x2_e4m3 fp8x2;
typedef __nv_fp8x4_e4m3 fp8x4;

void test_fp8_conversion()
{
    half a = 1.439; // Example fp8x2 value
    float b = 2.577f;

    fp8 a_fp8 = __nv_fp8_e4m3(a); // Convert half to fp8
    half a_fp8_half = static_cast<half>(a_fp8); // Convert fp8 to half
    printf("Converted half value: %f\n", __half2float(a_fp8_half)); // Print converted half value
    float a_fp8_half_float = static_cast<float>(a_fp8); // Convert fp8 to float
    printf("Converted fp8 value: %f\n", a_fp8_half_float); // Print converted fp8 value

    fp8 b_fp8 = __nv_fp8_e4m3(b); // Convert float to fp8
    half b_fp8_half = static_cast<half>(b_fp8); // Convert fp8 to half
    printf("Converted float value: %f\n", __half2float(b_fp8_half)); // Print converted float value
    
    // fp8 c = a_fp8 * b_fp8; // Add two fp8 values
    half2 c = {a, b}; // Create a half2 value
    printf("Half2 value: (%f, %f)\n", __half2float(c.x), __half2float(c.y)); // Print half2 value

    fp8x2 c_fp8x2 = __nv_fp8x2_e4m3(c); // Convert half2 to fp8x2
    half2 c_fp8x2_half = static_cast<half2>(c_fp8x2); // Convert fp8x2 to half2
    printf("Converted fp8x2 value: (%f, %f)\n", 
           __half2float(c_fp8x2_half.x), 
           __half2float(c_fp8x2_half.y)); 
    
}

__global__ void kernel(float *a, float *b, int m, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if(idx < m && idy < n) {
        // Convert float to fp8
        fp8 b_fp8 = __nv_fp8_e4m3(b[idx * n + idy]);
        // Convert fp8 to float
        float b_fp8_float = static_cast<float>(b_fp8);
        
        // b[idx * n + idy] = b_fp8_float; // Store the converted value back to b

        for(int i = 0; i < VEC; i++)
        {
            float a_reg = a[idx * n * VEC + idy * VEC + i];
            a[idx * n * VEC + idy * VEC + i] = a_reg * b_fp8_float; // Multiply each element of a by the converted b value
        }
    }
}

int main()
{
    int m = 8; // Number of rows
    int n = 1; // Number of columns

    float *h_a = (float *)malloc(m * n * VEC * sizeof(float));
    float *h_b = (float *)malloc(m * n * sizeof(float));
    // float *h_c = (float *)malloc(m * n * VEC * sizeof(float));

    // 初始化
    for (int i = 0; i < m * n * VEC; i++) {
        h_a[i] = 1.0f; 
    }
    for (int i = 0; i < m * n; i++) {
        // h_b[i] = static_cast<float>(rand()) / RAND_MAX;
        h_b[i] = (i+1) * 1.1111f; // For testing, use a constant value
    }

    // 打印输入
    // printf("Input matrix A:\n");
    // for (int i = 0; i < m; i++) {
    //     for (int j = 0; j < n * VEC; j++) {
    //         printf("%f ", h_a[i * n  * VEC + j]);
    //     }
    //     printf("\n");
    // }   
    // printf("Input matrix B:\n");
    // for (int i = 0; i < m; i++) {
    //     for (int j = 0; j < n; j++) {
    //         printf("%f ", h_b[i * n + j]);
    //     }
    //     printf("\n");
    // }

    float *d_a, *d_b;
    cudaMalloc((void **)&d_a, m * n * VEC * sizeof(float));
    cudaMalloc((void **)&d_b, m * n * sizeof(float));
    cudaMemcpy(d_a, h_a, m * n * VEC * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, m * n * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockDim(32, 1);
    dim3 gridDim((m + blockDim.x - 1) / blockDim.x, (n + blockDim.y - 1) / blockDim.y);
    #ifdef TEST_TIME
    // 创建CUDA事件计时
    // warm up
    for(int i = 0; i < 5 ; i++) {
        // Warm up
        kernel<<<gridDim, blockDim>>>(d_a, d_b, m, n);
    }
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for(int i = 0; i < ITERS ; i++) {
    #endif
    
        kernel<<<gridDim, blockDim>>>(d_a, d_b, m, n);   
    
    #ifdef TEST_TIME
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    printf("单次dequantize运行时间: %.12f ms\n", (float)(ms/ITERS));//
    #endif

    cudaDeviceSynchronize();
    cudaMemcpy(h_a, d_a, m * n * VEC * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_b, d_b, m * n * sizeof(float), cudaMemcpyDeviceToHost);

    // 打印输出
    // printf("Output matrix A:\n");
    // for (int i = 0; i < m; i++) {
    //     for (int j = 0; j < n * VEC; j++) {
    //         printf("%f ", h_a[i * n  * VEC + j]);
    //     }
    //     printf("\n");
    // }
    // printf("Output matrix B:\n");
    // for (int i = 0; i < m; i++) {
    //     for (int j = 0; j < n; j++) {
    //         printf("%f ", h_b[i * n + j]);
    //     }
    //     printf("\n");   
    // }

    // 释放内存
    free(h_a);
    free(h_b);
    cudaFree(d_a);
    cudaFree(d_b);
        
    // test_fp8_conversion();
    return 0;
}