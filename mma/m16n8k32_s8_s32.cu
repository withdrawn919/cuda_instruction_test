
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <cuda_runtime.h>

#include "../dequantize_fp4_fp8/wmma_utils.cuh"

#define CHECK_CUDA(status)                                                  \
{                                                                           \
    cudaError_t error = status;                                             \
    if (error != cudaSuccess) {                                             \
        std::cerr << "Got bad cuda status: " << cudaGetErrorString(error)   \
                << " at line: " << __LINE__ << std::endl;                   \
        exit(EXIT_FAILURE);                                                 \
    }                                                                       \
}

constexpr int WARP_SIZE = 32;
constexpr int M = 16;
constexpr int N = 8;  
constexpr int K = 32;

// 比较cpu和gpu的计算结果
bool compareArrays(int arr1[], int arr2[], int size) {
    for (int i = 0; i < size; ++i) {
        if (arr1[i] != arr2[i]) return false;
    }
    return true;
}


// CPU矩阵乘法: C[M][N] = A[M][K] * B[K][N]
void int8_gemm(const int8_t* A, const int8_t* B, int32_t* C, 
               int M, int N, int K) {
    // 初始化结果矩阵C为0
    for (int i = 0; i < M * N; ++i) {
        C[i] = 0;
    }

    // 三重循环计算矩阵乘法
    for (int i = 0; i < M; ++i) {          // 遍历A的行
        for (int j = 0; j < N; ++j) {      // 遍历B的列
            for (int k = 0; k < K; ++k) {  // 累加内积
                // 计算A[i][k] * B[k][j]并累加到C[i][j]
                C[i * N + j] += static_cast<int32_t>(A[i * K + k]) * 
                                static_cast<int32_t>(B[k * N + j]);
            }
        }
    }
}

struct SM80_16x8x32_S32S8S8S32_TN
{
  using DRegisters = uint32_t[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = uint32_t[4];

  __device__ static void
  fma(uint32_t      & d0, uint32_t      & d1, uint32_t      & d2, uint32_t      & d3,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      uint32_t const& c0, uint32_t const& c1, uint32_t const& c2, uint32_t const& c3)
  {
    asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5,  %6,  %7},"
      "{%8,  %9},"
      "{%10, %11, %12, %13};\n"
      : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
      :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
         "r"(b0),  "r"(b1),
         "r"(c0),  "r"(c1),  "r"(c2),  "r"(c3));
  }

  __device__ static void
  load_A_int8_G_row(const int8_t *A ,ARegisters &a_regs){
    const int laneid = threadIdx.x % 32;
    const int group_row = laneid >> 2; // laneid // 4
    const int group_col = laneid % 4; // laneid % 4
    
    a_regs[0] = *(int*)&A[group_row * 32 + group_col * 4];
    a_regs[1] = *(int*)&A[(group_row + 8) * 32 + group_col * 4];
    a_regs[2] = *(int*)&A[group_row * 32 + (group_col * 4 + 16)];
    a_regs[3] = *(int*)&A[(group_row + 8) * 32 + (group_col * 4 + 16)];

    if(laneid == 3){
        // printf("group_row_0:%d, group_col_0:%d\n",group_row,group_col * 4);
        // printf("group_row_1:%d, group_col_1:%d\n",group_row+8,group_col * 4);
        // printf("group_row_2:%d, group_col_2:%d\n",group_row,(group_col * 4+16) );
        // printf("group_row_3:%d, group_col_3:%d\n",group_row+8,(group_col * 4 +16));
        // printf("A[group_row * 32 + group_col * 4]:%d\n",A[group_row * 32 + group_col * 4]);
        // printf("A[(group_row + 8) * 32 + group_col * 4]:%d\n",A[(group_row + 8) * 32 + group_col * 4]);
        // printf("A[group_row * 32 + (group_col + 16) * 4]:%d\n",A[group_row * 32 + (group_col * 4 + 16)]);
        // printf("A[(group_row + 8) * 32 + (group_col + 16) * 4]:%d\n",A[(group_row + 8) * 32 + (group_col * 4 + 16)]);
        // print_binary_fp8(A[group_row * 32 + group_col * 4 + 0]);
        // print_binary_fp8(A[group_row * 32 + group_col * 4 + 1]);
        // print_binary_fp8(A[group_row * 32 + group_col * 4 + 2]);
        // print_binary_fp8(A[group_row * 32 + group_col * 4 + 3]);
        // print_binary32(a_regs[0]);
        printf("laneid:%d\n",laneid);
        printf("a0:%d\n",A[group_row * 32 + group_col * 4]);
        printf("a1:%d\n",A[group_row * 32 + group_col * 4 + 1]);
        printf("a2:%d\n",A[group_row * 32 + group_col * 4 + 2]);
        printf("a3:%d\n",A[group_row * 32 + group_col * 4 + 3]);
        printf("a4:%d\n",A[(group_row + 8) * 32 + group_col * 4]);
        printf("a5:%d\n",A[(group_row + 8) * 32 + group_col * 4 + 1]);
        printf("a6:%d\n",A[(group_row + 8) * 32 + group_col * 4 + 2]);
        printf("a7:%d\n",A[(group_row + 8) * 32 + group_col * 4 + 3]);
        printf("a8:%d\n",A[group_row * 32 + (group_col * 4 + 16)]);
        printf("a9:%d\n",A[group_row * 32 + (group_col * 4 + 16) + 1]);
        printf("a10:%d\n",A[group_row * 32 + (group_col * 4 + 16) + 2]);
        printf("a11:%d\n",A[group_row * 32 + (group_col * 4 + 16) + 3]);
        printf("a12:%d\n",A[(group_row + 8) * 32 + (group_col * 4 + 16)]);
        printf("a13:%d\n",A[(group_row + 8) * 32 + (group_col * 4 + 16) + 1]);
        printf("a14:%d\n",A[(group_row + 8) * 32 + (group_col * 4 + 16) + 2]);
        printf("a15:%d\n",A[(group_row + 8) * 32 + (group_col * 4 + 16) + 3]);
    }
  }

  __device__ static void
  load_A_int8_G_col(const int8_t *A ,ARegisters &a_regs){


  }

  __device__ static void
  load_B_int8_G_row(const int8_t *B ,BRegisters &b_regs){
    const int laneid = threadIdx.x % 32;
    const int groupID            = laneid >> 2; // laneid // 4
    const int threadID_in_group  = laneid % 4; // laneid % 4
    
    int8_t b0_7[8];
    
    int row;
    int col;
    #pragma unroll
    for(int i = 0; i < 8; i++){
        if(i < 4){
            row = (threadID_in_group * 4) + (i & 0x3);
        }else{
            row = (threadID_in_group * 4) + (i & 0x3) + 16;
        }
        col = groupID;
        b0_7[i] = B[row * 8 + col];
    } 
    b_regs[0] = (static_cast<uint8_t>(b0_7[3]) << 24) | 
                (static_cast<uint8_t>(b0_7[2]) << 16) | 
                (static_cast<uint8_t>(b0_7[1]) << 8)  | 
                 static_cast<uint8_t>(b0_7[0]);
    b_regs[1] = (static_cast<uint8_t>(b0_7[7]) << 24) | 
                (static_cast<uint8_t>(b0_7[6]) << 16) | 
                (static_cast<uint8_t>(b0_7[5]) << 8)  | 
                 static_cast<uint8_t>(b0_7[4]);
    if(laneid == 3){
        printf("laneid:%d\n",laneid);
        printf("b0_7[0]%d\n",b0_7[0]);
        printf("b0_7[1]%d\n",b0_7[1]);
        printf("b0_7[2]%d\n",b0_7[2]);
        printf("b0_7[3]%d\n",b0_7[3]);
        // print_binary_fp8(b0_7[0]);
        // print_binary_fp8(b0_7[1]);
        // print_binary_fp8(b0_7[2]);
        // print_binary_fp8(b0_7[3]);
        // print_binary32(b_regs[0]);

        printf("b0_7[4]%d\n",b0_7[4]);
        printf("b0_7[5]%d\n",b0_7[5]);
        printf("b0_7[6]%d\n",b0_7[6]);
        printf("b0_7[7]%d\n",b0_7[7]);
        // print_binary_fp8(b0_7[4]);
        // print_binary_fp8(b0_7[5]);
        // print_binary_fp8(b0_7[6]);
        // print_binary_fp8(b0_7[7]);
        // print_binary32(b_regs[1]);
    }
  }

  __device__ static void
  load_B_int8_G_col(const int8_t *B ,BRegisters &b_regs){
    const int laneid = threadIdx.x % 32;
    const int group_row = laneid >> 2; // laneid // 4
    const int group_col = laneid % 4; // laneid % 4

    b_regs[0] = *(int*)&B[group_row * 32 + group_col * 4];
    b_regs[1] = *(int*)&B[group_row * 32 + group_col * 4 + 16];

    if(laneid == -1){
        printf("B[group_row * 32 + group_col * 4]:%d\n",B[group_row * 32 + group_col * 4]);
        printf("B[group_row * 32 + group_col * 4 + 1]:%d\n",B[group_row * 32 + group_col * 4 + 1]);
        printf("B[group_row * 32 + group_col * 4 + 2]:%d\n",B[group_row * 32 + group_col * 4 + 2]);
        printf("B[group_row * 32 + group_col * 4 + 3]:%d\n",B[group_row * 32 + group_col * 4 + 3]);
        print_binary_fp8(B[group_row * 32 + group_col * 4]);
        print_binary_fp8(B[group_row * 32 + group_col * 4 +1]);
        print_binary_fp8(B[group_row * 32 + group_col * 4 + 2]);
        print_binary_fp8(B[group_row * 32 + group_col * 4 + 3]);
        print_binary32(b_regs[0]);
    }
  }
  __device__ static void
    store_D(int *D,DRegisters &d_regs){
        const int laneid = threadIdx.x % 32;
        const int group_row = laneid >> 2;
        const int group_col = laneid % 4;
        D[group_row * 8 + group_col * 2] = d_regs[0];
        D[group_row * 8 + group_col * 2 + 1] = d_regs[1];
        D[(group_row + 8) * 8 + group_col * 2] = d_regs[2];
        D[(group_row + 8) * 8 + group_col * 2 + 1] = d_regs[3];
        if(laneid == 3){
            printf("laneid:%d\n",laneid);
            printf("D[%d][%d]=%d\n",group_row,group_col * 2,d_regs[0]);
            printf("D[%d][%d]=%d\n",group_row,group_col * 2 + 1,d_regs[1]);
            printf("D[%d][%d]=%d\n",group_row + 8,group_col * 2,d_regs[2]);
            printf("D[%d][%d]=%d\n",group_row + 8,group_col * 2 + 1,d_regs[3]);
        }
    }
};

__global__ void mma_SM80_16x8x32_S32S8S8S32_TN(int8_t* input_A, int8_t* input_B, int* input_C, int* output_D){
   
    SM80_16x8x32_S32S8S8S32_TN::ARegisters a_regs;
    SM80_16x8x32_S32S8S8S32_TN::load_A_int8_G_row(input_A, a_regs);
    SM80_16x8x32_S32S8S8S32_TN::BRegisters b_regs;
    SM80_16x8x32_S32S8S8S32_TN::load_B_int8_G_row(input_B, b_regs);
    // SM80_16x8x32_S32S8S8S32_TN::load_B_int8_G_col(input_B, b_regs);
    SM80_16x8x32_S32S8S8S32_TN::CRegisters c_regs = {0,0,0,0};
    SM80_16x8x32_S32S8S8S32_TN::DRegisters d_regs = {0,0,0,0};

    SM80_16x8x32_S32S8S8S32_TN::fma(d_regs[0], d_regs[1], d_regs[2], d_regs[3],
                                    a_regs[0], a_regs[1], a_regs[2], a_regs[3],
                                    b_regs[0], b_regs[1],
                                    c_regs[0], c_regs[1], c_regs[2], c_regs[3]);
    SM80_16x8x32_S32S8S8S32_TN::store_D(output_D, d_regs);
}

int main(){
    
    int8_t      *host_a = new int8_t[M*K];
    int8_t      *host_b = new int8_t[K*N];
    int         *host_c = new int[K*N];
    int         *host_d = new int[M*N];

    int8_t      *dev_a;
    int8_t      *dev_b;
    int         *dev_c;
    int         *dev_d;

    CHECK_CUDA(cudaMalloc(&dev_a, sizeof(int8_t)*M*K));
    CHECK_CUDA(cudaMalloc(&dev_b, sizeof(int8_t)*K*N));
    CHECK_CUDA(cudaMalloc(&dev_c, sizeof(int)*K*N));
    CHECK_CUDA(cudaMalloc(&dev_d, sizeof(int)*M*N));
    // 主机数据初始化
    // A矩阵为对角矩阵
    for(int i=0;i<M;i++){
        for(int j=0;j<K;j++){
            if(i == j)
                host_a[i*K+j] = static_cast<int8_t>(1.0f);
            else
                host_a[i*K+j] = static_cast<int8_t>(0.0f);
        }
    }
    for(int i = 0; i < M*K; ++i)    host_a[i] = (i*1.0f); 
    // B矩阵行主序递增
    for(int i = 0; i < K*N; ++i)    host_b[i] =(i*1.0f); 

    // C矩阵为单位对角矩阵
    for(int i=0;i<K;i++){
        for(int j=0;j<N;j++){
            if(i == j)
                host_c[i*N+j] = static_cast<int>(1.0f);
            else
                host_c[i*N+j] = static_cast<int>(0.0f);
        }
    }
    for(int i = 0; i < K*N; ++i)    host_c[i] = 0.0f;

    for(int i = 0; i < M*N; ++i)    host_d[i] = 0.0f;
    // 打印一下

    printf("host_a:\n");
    dumpEx_row_major<int8_t>(host_a, M, K,"host_a",8,4);
    printf("host_b:\n");
    // dumpEx_col_major<int8_t>(host_b, K, N,"host_b");
    dumpEx_row_major<int8_t>(host_b, K, N,"host_b",4,1);
    printf("host_c:\n");
    dumpEx_row_major<int>(host_c, K, N,"host_c",8,2);

    CHECK_CUDA(cudaMemcpy(dev_a, host_a, sizeof(int8_t)*M*K, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dev_b, host_b, sizeof(int8_t)*K*N, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dev_c, host_c, sizeof(int)*K*N, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dev_d, host_d, sizeof(int)*M*N, cudaMemcpyHostToDevice));

    dim3 block(32);
    dim3 grid(1);

    mma_SM80_16x8x32_S32S8S8S32_TN<<<grid,block>>>(dev_a, dev_b, dev_c, dev_d);
    cudaDeviceSynchronize();

    CHECK_CUDA(cudaMemcpy(host_d, dev_d, sizeof(int)*M*N, cudaMemcpyDeviceToHost));
    dumpEx_row_major<int>(host_d, M, N, "host_d",8,2);

    // cpu reference
    int8_gemm(host_a, host_b, host_c, M, N, K);
    dumpEx_row_major<int>(host_c, M, N, "host_c_ref");

    if(compareArrays(host_c, host_d, M*N)){
        std::cout << "Results match!" << std::endl;
    } else {
        std::cout << "Results do not match!" << std::endl;
    }

    CHECK_CUDA(cudaFree(dev_a));
    CHECK_CUDA(cudaFree(dev_b));
    CHECK_CUDA(cudaFree(dev_c));
    CHECK_CUDA(cudaFree(dev_d));
    delete[] host_a;
    delete[] host_b;
    delete[] host_c;
    delete[] host_d; 
    return 0;
}