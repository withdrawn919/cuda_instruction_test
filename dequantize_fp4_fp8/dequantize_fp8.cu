#include <cstdio>
#include <cuda_fp8.h>
#include <../config.h>

using fp8e4m3 = __nv_fp8_e4m3;
using fp32 = float;


__global__ void dequantize(fp8e4m3* input, fp32* output, fp32 scale_inv) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    constexpr int nvec = 32 / sizeof(fp32);
    for (int i = 0; i < nvec; ++i) {
        // 1.读取fp8
        fp8e4m3 inp = input[tid * 8 + i];
        // 2.fp8转fp32
        const fp32 val = static_cast<fp32>(inp);
        // 3.反量化
        fp32 temp = val * (scale_inv);
        // 4.写回
        output[tid * 8 + i] = temp;
    }
}

int main() {
    fp8e4m3 h_input[16][8];
    fp32 h_output[16][8];
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 8; j++) {
            h_input[i][j] = static_cast<fp8e4m3>(0.1 * (i * 16 + j + 1));
        }
    }

    fp8e4m3* d_input = nullptr;
    fp32* d_output = nullptr;
    fp32 scale_inv = 1.1f;

    cudaMalloc(&d_input, sizeof(fp8e4m3) * 16 * 8);
    cudaMalloc(&d_output, sizeof(fp32) * 16 * 8);
    cudaMemcpy(d_input, h_input, sizeof(fp8e4m3) * 16 * 8, cudaMemcpyHostToDevice);

    #ifdef TEST_TIME
    // 创建CUDA事件计时
    // warm up
    for(int i = 0; i < ITERS ; i++) {
        // Warm up
        dequantize<<<1, 16>>>(d_input, d_output, scale_inv);
    }
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for(int i = 0; i < ITERS ; i++) {
    #endif

    dequantize<<<1, 16>>>(d_input, d_output, scale_inv);
    
    #ifdef TEST_TIME
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    printf("单次dequantize运行时间: %.12f ms\n", (float)(ms/ITERS));//
    #endif
    
    cudaMemcpy(h_output, d_output, sizeof(fp32) * 16 * 8, cudaMemcpyDeviceToHost);

    return 0;
}