#include <cassert>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cstdint>
#include <cuda_fp4.h>

#include "../config.h"


using fp4 = __nv_fp4_e2m1;


struct FP4_SM120_16x8x32_TN {
        using DRegisters = float[4];
        using ARegisters = uint32_t[4];
        using BRegisters = uint32_t[2];
        using CRegisters = float[4];
   
    __device__ static void fma(float &d0, float &d1, float &d2, float &d3,
                               uint32_t const &a0, uint32_t const &a1,
                               uint32_t const &a2, uint32_t const &a3,
                               uint32_t const &b0, uint32_t const &b1,
                               float const &c0, float const &c1,
                               float const &c2, float const &c3) {
                  asm volatile(
                    "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e2m1.e2m1.f32 "
                    "{%0,  %1,  %2,  %3},"
                    "{%4,  %5,  %6,  %7},"
                    "{%8,  %9},"
                    "{%10, %11, %12, %13};\n"
                    : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(c0),
                      "f"(c1), "f"(c2), "f"(c3));
    }
    __device__ static void
    load_A_float_G(const fp4 *A ,ARegisters &a_regs){

    }

};


  __global__ void fp4_mma_kernel(
    float* d_output,          // 输出D寄存器结果
    const uint32_t* a_data,   // A寄存器数据指针
    const uint32_t* b_data,   // B寄存器数据指针
    const float* c_data       // C寄存器数据指针
) {

    FP4_SM120_16x8x32_TN::fma(
        d_output[0], d_output[1], d_output[2], d_output[3],
        a_data[0], a_data[1], a_data[2], a_data[3],
        b_data[0], b_data[1],
        c_data[0], c_data[1], c_data[2], c_data[3]
    );

}

int main(){
    uint8_t num = 0b00000110; // 测试值
    fp4 fp4_num = *reinterpret_cast<fp4*>(&num); // 测试值
    fp4_num = __nv_fp4_e2m1(0.25f); // 测试值转换为fp4类型
    fp4 fp4_min = __nv_fp4_e2m1(0b0000); // 最小值
    fp4 fp4_zero = __nv_fp4_e2m1(0b0000); // 零值
    fp4 fp4_one = __nv_fp4_e2m1(0b0001); // 单位
    printf("fp4_each_bit: 0b%d%d%d%d%d%d%d%d\n", 
           (char)fp4_num & 0b10000000 ? 1 : 0,
           (char)fp4_num & 0b01000000 ? 1 : 0,
           (char)fp4_num & 0b00100000 ? 1 : 0,
           (char)fp4_num & 0b00010000 ? 1 : 0,
           (char)fp4_num & 0b00001000 ? 1 : 0,
           (char)fp4_num & 0b00000100 ? 1 : 0,
           (char)fp4_num & 0b00000010 ? 1 : 0,
           (char)fp4_num & 0b00000001 ? 1 : 0);
    printf("fp4_sizeof: %zu\n", sizeof(fp4));
    printf("fp4_mnum: %f\n", static_cast<float>(fp4_num));
    printf("fp4_min: %f\n", static_cast<float>(fp4_min));
    printf("fp4_zero: %f\n", static_cast<float>(fp4_zero));
    printf("fp4_one: %f\n", static_cast<float>(fp4_one));
    

    // 主机端数据初始化 D=A×B+C
    float h_d[4]   = {0.0f, 0.0f, 0.0f, 0.0f};
    uint32_t h_a[4]     = {0x06060606, 0x06060606, 0x06060606, 0x06060606};
    uint32_t h_b[2]     = {0x06060606, 0x06060606};
    float h_c[4]        = {0.0f, 0.0f, 0.0f, 0.0f};

    // 设备端内存分配
    float* d_d;
    uint32_t* d_a;
    uint32_t* d_b;
    float* d_c;

    cudaMalloc(&d_d, 4 * sizeof(float));
    cudaMalloc(&d_a, 4 * sizeof(uint32_t));
    cudaMalloc(&d_b, 2 * sizeof(uint32_t));
    cudaMalloc(&d_c, 4 * sizeof(float));

    cudaMemcpy(d_d, h_d, 4 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_a, h_a, 4 * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, 2 * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, h_c, 4 * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(32);
    dim3 grid(1);

    #ifdef TEST_TIME
    // 创建CUDA事件计时
    // warm up
    // for(int i = 0; i < ITERS ; i++) {
        // Warm up
        fp4_mma_kernel<<<grid,block>>>(d_d,d_a,d_b,d_c);
    // }
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for(int i = 0; i < ITERS ; i++) {
    #endif

    fp4_mma_kernel<<<grid,block>>>(d_d,d_a,d_b,d_c);
    
    #ifdef TEST_TIME
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    printf("单次mma_fp4的Kernel运行时间: %.12f ms\n", (float)(ms/ITERS));//
    #endif
    cudaDeviceSynchronize(); 


    cudaMemcpy(h_d, d_d, 4 * sizeof(float), cudaMemcpyDeviceToHost);
    printf("h_d:\n");
    for(int i = 0; i < 4; i++) {
        printf("%f ", h_d[i]);
    }
    printf("\n");
    
    // 资源释放
    cudaFree(d_d); cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

    #ifdef TEST_TIME
    // 资源释放
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    #endif

    return 0;
}