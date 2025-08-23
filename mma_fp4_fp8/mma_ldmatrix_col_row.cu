#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <stdlib.h>
#include <cuda_fp16.h>          // FP16 支持
#include <mma.h>          // WMMA API

using namespace nvcuda;

// 定义Tensor Core配置
#define WMMA_M 32
#define WMMA_N 8
#define WMMA_K 16

template<typename T>
__device__ __host__
void dumpEx1(T* data, const int M, const int N, const char* name, 
            int group_row = 0, int group_col = 0) {
    printf("Dumping %s (Shape: %d x %d):\n", name, M, N);
    printf("     "); // 对齐列号标题
    
    // 打印列号标题 + 按列分组分隔符
    for (int c = 0; c < N; c++) {
        if (group_col > 0 && c % group_col == 0 && c != 0) 
            printf(" | "); // 列分组分隔符
        printf("%8d ", c); // 列号
    }
    printf("\n");
    
    // 打印数据行（按行分组）
    for (int r = 0; r < M; r++) {
        // 行分组分隔符
        if (group_row > 0 && r % group_row == 0 && r != 0) {
            printf("----");
            for (int c = 0; c < N; c++) {
                if (group_col > 0 && c % group_col == 0 && c != 0) 
                    printf("----------------");
                printf("---------");
            }
            printf("\n");
        }
        
        printf("%3d: ", r); // 行号标题
        for (int c = 0; c < N; c++) {
            // 列分组分隔符
            if (group_col > 0 && c % group_col == 0 && c != 0) 
                printf(" | ");
            
            // 打印数据（保留3位小数）
            printf("%8.3f ", static_cast<float>(data[r * N + c]));
        }
        printf("\n");
    }
}

// CPU参考模型实现（与原代码一致）
void gemm_cpu(const std::vector<float>& A, const std::vector<float>& B,
              std::vector<float>& C_ref, int M, int N, int K,
              float alpha, float beta) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[m * K + k] * B[k * N + n];
            }
            C_ref[m * N + n] = alpha * sum + beta * C_ref[m * N + n];
        }
    }
}

// Tensor Core GEMM kernel（FP16 输入，FP32 输出）
__global__ void tensorCoreGemm(half* A, half* B, float* C, int M, int N, int K) {
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    // int row = blockIdx.y * blockDim.y + threadIdx.y;
    // int col = blockIdx.x * blockDim.x + threadIdx.x;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int inner_k_loop = K / WMMA_K;
    int a_row = by * WMMA_M;
    int b_col = bx * WMMA_N;

    for (int k = 0; k < inner_k_loop; k++) {
        wmma::load_matrix_sync(a_frag, &A[a_row * K + k * WMMA_K], K);
        wmma::load_matrix_sync(b_frag, &B[k * N * WMMA_K + b_col], N);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    wmma::store_matrix_sync(&C[a_row * N + b_col], c_frag, N, wmma::mem_row_major);
}

void printMatrix(float* mat, int M, int N) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << mat[i * N + j] << " ";
        }
        std::cout << std::endl;
    }
}

int main(int argc, char* argv[]) { 
    // 初始化矩阵（FP16 随机值）
    int M = 32, N = 8, K = 16;
    std::vector<half> A(M * K), B(K * N);
    std::vector<float> C_single(M * N), C_dual(M * N), C_ref(M * N);

    for (auto& x : A) x = __float2half(static_cast<float>(rand()) / RAND_MAX);
    for (auto& x : B) x = __float2half(static_cast<float>(rand()) / RAND_MAX);

    // printf("Matrix A (M x K):\n");
    // printMatrix(reinterpret_cast<float*>(A.data()), M, K);
    // printf("Matrix B (K x N):\n");
    // printMatrix(reinterpret_cast<float*>(B.data()), K, N);

    float alpha = 1.0f, beta = 0.0f;
    int warmup = 5;
    // int loops = 50;

    // CPU 参考模型
    std::vector<float> A_float(M * K);
    for (int i = 0; i < M * K; ++i) A_float[i] = __half2float(A[i]);
    std::vector<float> B_float(K * N);
    for (int i = 0; i < K * N; ++i) B_float[i] = __half2float(B[i]);
    gemm_cpu(A_float, B_float, C_ref, M, N, K, alpha, beta);
    printf("Matrix C_ref (M x N):\n");
    // printMatrix(C_ref.data(), M, N);
    dumpEx1<float>(C_ref.data(), M, N, "C_ref");

    half* A_d; half* B_d; float* C_single_d;
    cudaMalloc(&A_d, M * K * sizeof(half));
    cudaMalloc(&B_d, K * N * sizeof(half));
    cudaMalloc(&C_single_d, M * N * sizeof(float));
    cudaMemcpy(A_d, A.data(), M * K * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B.data(), K * N * sizeof(half), cudaMemcpyHostToDevice);

    dim3 threads(32, 1);
    dim3 blocks((M + WMMA_M - 1) / WMMA_M, (N + WMMA_N - 1) / WMMA_N);

    // 预热
    for (int i = 0; i < warmup; ++i) {
        tensorCoreGemm<<<blocks, threads>>>(A_d, B_d, C_single_d, M, N, K);
    }
    cudaDeviceSynchronize();
    cudaMemcpy(C_single.data(), C_single_d, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    printf("Matrix C_single (M x N):\n");
    // printMatrix(C_single.data(), M, N);
    dumpEx1<float>(C_single.data(), M, N, "C_single");

    // 验证单卡结果
    float max_error = 0.0f;
    for (int i = 0; i < M * N; ++i) {
        float diff = fabs(C_single[i] - C_ref[i]);
        if (diff > max_error) max_error = diff;
    }
    std::cout << "Max error (Single GPU vs CPU): " << max_error << std::endl;
    return 0;
}