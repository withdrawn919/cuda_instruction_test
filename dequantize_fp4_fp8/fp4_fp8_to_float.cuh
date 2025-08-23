/*
    封装了两个PTX：
    1. 将4个e4m3x2打包的uint32_t转换为float4
    2. 将8个e2m1x2打包的uint32_t转换为float*，
       其中float*的长度为8，存储8个float值
*/


#include <cuda_runtime.h>
#include <iostream>
#include <stdlib.h>

#include <cuda_runtime.h>
#include <iostream>
#include <stdlib.h>
#include <random>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>


// 4个fp8打包一个uint32_t
// 其中每个fp8占用8位
__device__ float4 convert_e4m3x4_to_float4(uint32_t const & src_packed){
    
    uint32_t out_fp16[2];// 四个half
    // 使用 PTX 指令进行转换
    asm volatile(
        "{\n"
        "    .reg.b16 lo, hi;\n"
        "    mov.b32 {lo, hi}, %2;"
        "    cvt.rn.f16x2.e4m3x2 %0, lo;\n" // 将 e4m3x2 转换为 f16x2
        "    cvt.rn.f16x2.e4m3x2 %1, hi;\n"
        "}\n"
        : "=r"(out_fp16[0]),"=r"(out_fp16[1]) // 输出为 64 位寄存器，存储两个半精度浮点数
        : "r"(src_packed)                            // 输入为 32 位寄存器，存储 e4m3x2 数据
    );
    float2 res0 = __half22float2(reinterpret_cast<__half2 &>(out_fp16[0]));
    float2 res1 = __half22float2(reinterpret_cast<__half2 &>(out_fp16[1]));

    float4 out;
    out.x = res0.x;
    out.y = res0.y;
    out.z = res1.x;
    out.w = res1.y;
    // printf("out: %f, %f, %f, %f\n", out.x, out.y, out.z, out.w);
    return out;
}

// 8个fp4打包一个uint32_t
// 其中每个fp4占用4位
__device__ float* convert_e2m1x8_to_float(uint32_t const & src_packed){
    
    uint32_t out_fp16[4];// 八个half
    // 使用 PTX 指令进行转换
    asm volatile(
        "{\n"
        "    .reg.b8 byte0, byte1, byte2, byte3;\n"
        "    mov.b32 {byte0, byte1, byte2, byte3}, %4;"
        "    cvt.rn.f16x2.e2m1x2 %0, byte0;\n" // 将 e2m1x2 转换为 f16x2
        "    cvt.rn.f16x2.e2m1x2 %1, byte1;\n"
        "    cvt.rn.f16x2.e2m1x2 %1, byte2;\n"
        "    cvt.rn.f16x2.e2m1x2 %1, byte3;\n"
        "}\n"
        : "=r"(out_fp16[0]),"=r"(out_fp16[1]),"=r"(out_fp16[2]),"=r"(out_fp16[3]) // 输出为 64 位寄存器，存储两个半精度浮点数
        : "r"(src_packed)                            // 输入为 32 位寄存器，存储 e2m1x2 数据
    );
    float2 res0 = __half22float2(reinterpret_cast<__half2 &>(out_fp16[0]));
    float2 res1 = __half22float2(reinterpret_cast<__half2 &>(out_fp16[1]));
    float2 res2 = __half22float2(reinterpret_cast<__half2 &>(out_fp16[2]));
    float2 res3 = __half22float2(reinterpret_cast<__half2 &>(out_fp16[3]));

    float* out = new float[8];
    out[0] = res0.x;
    out[1] = res0.y;
    out[2] = res1.x;
    out[3] = res1.y;
    out[4] = res2.x;
    out[5] = res2.y;
    out[6] = res3.x;
    out[7] = res3.y;
    return out;
}