
#include <cuda_runtime.h>
#include <stdio.h>
#include <cstdint>

#include "../config.h"

struct FP4_SF_SM120_16x8x32_TN {
        using DRegisters = float[4];
        using ARegisters = uint32_t[4];
        using BRegisters = uint32_t[2];
        using CRegisters = float[4];

        using SFARegisters = uint8_t[1];
        using SFBRegisters = uint8_t[1];

        static constexpr uint16_t tidA = 0;
        static constexpr uint16_t bidA = 0;
        static constexpr uint16_t tidB = 0;
        static constexpr uint16_t bidB = 0;
   
    __device__ static void fma(float &d0, float &d1, float &d2, float &d3,
                               uint32_t const &a0, uint32_t const &a1,
                               uint32_t const &a2, uint32_t const &a3,
                               uint32_t const &b0, uint32_t const &b1,
                               float const &c0, float const &c1,
                               float const &c2, float const &c3,
                               uint8_t const &sfa0, uint8_t const &sfb0) {

                                asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X."
                                    "m16n8k32.row.col.f32.e2m1.e2m1.f32.ue8m0 "
                                    "{%0,  %1,  %2,  %3},"
                                    "{%4,  %5,  %6,  %7},"
                                    "{%8,  %9},"
                                    "{%10, %11, %12, %13},"
                                    "{%14},"
                                    "{%15, %16},"
                                    "{%17},"
                                    "{%18, %19};\n"
                                    : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                                      "f"(c0), "f"(c1), "f"(c2), "f"(c3), "r"(uint32_t(sfa0)),
                                      "h"(bidA), "h"(tidA), "r"(uint32_t(sfb0)), "h"(bidB),
                                      "h"(tidB));
                      
    }

};


  __global__ void fp4_sf_mma_kernel(
    float* d_output,          // 输出D寄存器结果
    const uint32_t* a_data,   // A寄存器数据指针
    const uint32_t* b_data,   // B寄存器数据指针
    const float* c_data,      // C寄存器数据指针
    const uint8_t* a_sf,
    const uint8_t* b_sf
) {

    FP4_SF_SM120_16x8x32_TN::fma(
        d_output[0], d_output[1], d_output[2], d_output[3],
        a_data[0], a_data[1], a_data[2], a_data[3],
        b_data[0], b_data[1],
        c_data[0], c_data[1], c_data[2], c_data[3],
        a_sf[0],b_sf[0]
    );
}

int main(){
    // 主机端数据初始化 D=A×B+C
    float h_d[4]   = {0.0f, 0.0f, 0.0f, 0.0f};
    uint32_t h_a[4]     = {0x22222222, 0x22222222, 0x22222222, 0x22222222};
    uint32_t h_b[2]     = {0x22222222, 0x22222222};
    float h_c[4]        = {0.0f, 0.0f, 0.0f, 0.0f};
    uint8_t h_sfa[1]    = {1};
    uint8_t h_sfb[1]    = {1};


    // 设备端内存分配
    float* d_d;
    uint32_t* d_a;
    uint32_t* d_b;
    float* d_c;
    uint8_t* d_sfa;  
    uint8_t* d_sfb;

    cudaMalloc(&d_d, 4 * sizeof(float));
    cudaMalloc(&d_a, 4 * sizeof(uint32_t));
    cudaMalloc(&d_b, 2 * sizeof(uint32_t));
    cudaMalloc(&d_c, 4 * sizeof(float));
    cudaMalloc(&d_sfa, 4 * sizeof(uint8_t));
    cudaMalloc(&d_sfb, 4 * sizeof(uint8_t));

    cudaMemcpy(d_d, h_d, 4 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_a, h_a, 4 * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, 2 * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, h_c, 4 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sfa, h_sfa, sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sfb, h_sfb, sizeof(uint8_t), cudaMemcpyHostToDevice);


    dim3 block(32);
    dim3 grid(1);

    #ifdef TEST_TIME
    // 创建CUDA事件计时
    // warm up
    for(int i = 0; i < ITERS ; i++) {
        // Warm up
        fp4_sf_mma_kernel<<<grid,block>>>(d_d,d_a,d_b,d_c,d_sfa, d_sfb);
    }
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for(int i = 0; i < ITERS ; i++) {
    #endif

    fp4_sf_mma_kernel<<<grid,block>>>(d_d,d_a,d_b,d_c,d_sfa, d_sfb);
    
    #ifdef TEST_TIME
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    printf("单次mma_fp4_sf的Kernel运行时间: %.12f ms\n", (float)(ms/ITERS));//
    #endif
    cudaDeviceSynchronize(); 
    

    cudaMemcpy(h_d, d_d, 4 * sizeof(float), cudaMemcpyDeviceToHost);
    
    // 资源释放
    cudaFree(d_d); cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    cudaFree(d_sfa);cudaFree(d_sfb);
    #ifdef TEST_TIME
    // 资源释放
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    #endif

    return 0;
}