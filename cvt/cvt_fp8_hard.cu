#include <cuda_runtime.h>
#include <cuda_fp8.h>  // FP8支持头文件
#include <iostream>
#include <vector>       // 添加vector头文件用于性能测试
#include <cmath>        // 添加cmath头文件用于pow函数

// 核函数：将float数组转换为FP8格式
__global__ void float_to_fp8_kernel(
    const float* input,    // 输入float数组
    __nv_fp8_storage_t* output,  // 输出FP8数组
    int size,              // 数据长度
    __nv_saturation_t saturate,  // 饱和模式
    __nv_fp8_interpretation_t fp8_type  // FP8格式类型
) {
    // 计算全局线程索引
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 确保索引在有效范围内
    if (idx < size) {
        // 核心转换调用
        output[idx] = __nv_cvt_float_to_fp8(
            input[idx],   // 输入浮点值
            saturate,     // 饱和处理模式
            fp8_type      // FP8格式类型（E4M3或E5M2）
        );
    }
}

int main() {
    const int N = 10000;  // 测试数据量
    // 1. 初始化主机数据
    float* h_input = new float[N];
    for (int i = 0; i < N; ++i) {
        h_input[i] = static_cast<float>(i) * 0.1f;  // 生成递增序列
    }

    // 2. 分配设备内存
    float* d_input;
    __nv_fp8_storage_t* d_output;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(__nv_fp8_storage_t));
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

    // 3. 配置核函数参数
    const int blockSize = 256;
    const int gridSize = (N + blockSize - 1) / blockSize;
    const auto saturate = __NV_SATFINITE;  // 启用饱和截断
    const auto fp8_type = __NV_E4M3;       // 使用E4M3格式

    // 4. 启动核函数
    float_to_fp8_kernel<<<gridSize, blockSize>>>(
        d_input, d_output, N, saturate, fp8_type
    );
    
    // 5. 回传结果并验证
    __nv_fp8_storage_t* h_output = new __nv_fp8_storage_t[N];
    cudaMemcpy(h_output, d_output, N * sizeof(__nv_fp8_storage_t), cudaMemcpyDeviceToHost);

    // 验证前5个结果
    std::cout << "FP8转换验证:\n";
    for (int i = 0; i < 5; ++i) {
        printf("├─ 输入: %.2f → FP8: 0x%02X\n", 
               h_input[i], 
               static_cast<unsigned>(h_output[i]));
    }

    // 6. 性能测试（添加的性能报告部分）[6,9](@ref)
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    const int warmup_runs = 10;  // 预热次数
    const int timed_runs = 10000;   // 正式计时次数
    
    // 预热GPU
    for (int i = 0; i < warmup_runs; ++i) {
        float_to_fp8_kernel<<<gridSize, blockSize>>>(d_input, d_output, N, saturate, fp8_type);
    }
    
    // 正式计时
    cudaEventRecord(start);
    for (int i = 0; i < timed_runs; ++i) {
        float_to_fp8_kernel<<<gridSize, blockSize>>>(d_input, d_output, N, saturate, fp8_type);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    // 计算性能指标
    float total_ms = 0;
    cudaEventElapsedTime(&total_ms, start, stop);
    float avg_ms_per_run = total_ms / timed_runs;
    float throughput = (N * timed_runs) / (total_ms / 1000) / 1e9;  // G元素/秒
    
    // 输出性能报告
    std::cout << "\n性能报告:\n";
    std::cout << "├─ 测试数据量: " << N << " 元素\n";
    std::cout << "├─ 总运行次数: " << timed_runs << "\n";
    std::cout << "├─ 总耗时: " << total_ms << " 毫秒\n";
    std::cout << "├─ 平均转换耗时: " << avg_ms_per_run << " 毫秒\n";
    std::cout << "├─ 吞吐量: " << throughput << " G元素/秒\n";
    std::cout << "└─ 注: Hopper架构(SM90+)有FP8硬件加速，性能提升显著[9](@ref)\n";

    // 7. 资源清理
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    delete[] h_input;
    delete[] h_output;
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}