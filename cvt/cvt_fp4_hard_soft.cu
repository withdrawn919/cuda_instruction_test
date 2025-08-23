#include <cuda_runtime.h>
#include <cuda_fp4.h>      // FP4支持头文件
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>         // 添加assert支持

// 核函数：将float数组转换为FP4格式
__global__ void float_to_fp4_kernel(
    const float* input,           // 输入float数组
    __nv_fp4_storage_t* output,    // 输出FP4数组
    int size,                     // 数据长度
    __nv_fp4_interpretation_t fp4_type, // FP4格式类型
    cudaRoundMode rounding        // 舍入模式
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // 核心转换调用
        output[idx] = __nv_cvt_float_to_fp4(
            input[idx],   // 输入浮点值
            fp4_type,     // FP4格式类型
            rounding      // 舍入模式
        );
    }
}

int main() {
    const int N = 10000;  // 测试数据量
    // 1. 初始化主机数据（使用小范围值避免FP4溢出）
    std::vector<float> h_input(N);
    for (int i = 0; i < N; ++i) {
        // FP4范围有限(-6.0~6.0)，生成-5.0到5.0的数据
        h_input[i] = (i % 100) * 0.1f - 5.0f; 
    }

    // 2. 分配设备内存
    float* d_input;
    __nv_fp4_storage_t* d_output;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(__nv_fp4_storage_t));
    cudaMemcpy(d_input, h_input.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    // 3. 配置核函数参数
    const int blockSize = 256;
    const int gridSize = (N + blockSize - 1) / blockSize;
    const auto fp4_type = __NV_E2M1;       // 使用E2M1格式
    const cudaRoundMode rounding_modes[] = {cudaRoundNearest, cudaRoundZero};

    // 4. 执行两种舍入模式的转换
    for (auto rounding : rounding_modes) {
        std::cout << "\n=== 舍入模式: " 
                  << (rounding == cudaRoundNearest ? "最近偶数" : "向零舍入") 
                  << " ===\n";
        
        // 启动核函数
        float_to_fp4_kernel<<<gridSize, blockSize>>>(d_input, d_output, N, fp4_type, rounding);
        cudaDeviceSynchronize();

        // 回传结果
        std::vector<__nv_fp4_storage_t> h_output(N);
        cudaMemcpy(h_output.data(), d_output, N * sizeof(__nv_fp4_storage_t), cudaMemcpyDeviceToHost);

        // 验证转换结果
        std::cout << "前5个转换结果:\n";
        for (int i = 0; i < 5; ++i) {
            // FP4值解码（实际应用中需实现FP4转float）
            printf("├─ 输入: %+6.2f → FP4: 0x%X\n", 
                   h_input[i], 
                   static_cast<unsigned>(h_output[i]) & 0xF); // 取低4位
        }

        // 5. 性能测试
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        const int warmup_runs = 50;
        const int timed_runs = 10000;
        
        // 预热
        for (int i = 0; i < warmup_runs; ++i) {
            float_to_fp4_kernel<<<gridSize, blockSize>>>(d_input, d_output, N, fp4_type, rounding);
        }
        
        // 正式计时
        cudaEventRecord(start);
        for (int i = 0; i < timed_runs; ++i) {
            float_to_fp4_kernel<<<gridSize, blockSize>>>(d_input, d_output, N, fp4_type, rounding);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float total_ms = 0;
        cudaEventElapsedTime(&total_ms, start, stop);
        float avg_ms_per_run = total_ms / timed_runs;
        float throughput = (N * timed_runs) / (total_ms / 1000) / 1e9;  // G元素/秒
        
        // 输出性能报告
        std::cout << "\n性能报告:\n";
        std::cout << "├─ 数据量: " << N << " 元素\n";
        std::cout << "├─ 运行次数: " << timed_runs << "\n";
        std::cout << "├─ 总耗时: " << total_ms << " ms\n";
        std::cout << "├─ 平均转换耗时: " << avg_ms_per_run << " 毫秒\n";
        std::cout << "├─ 吞吐量: " << throughput << " G元素/秒\n";
        std::cout << "└─ 注: 在Blackwell架构(SM90+)上可启用FP4硬件加速[9,11](@ref)\n";

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    // 6. 资源清理
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}