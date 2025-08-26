#include <cstdint>
#include <cstdio>
#include <iostream>
#include <cuda_runtime.h>
#include <sys/types.h>
#include <vector>
#include <random>

#include "../dequantize_fp4_fp8/wmma_utils.cuh"

constexpr int WARP_SIZE = 32;
constexpr int M = 16;
constexpr int N = 8;  
constexpr int K = 32;

// 比较cpu和gpu的计算结果
bool compareArrays(float arr1[], float arr2[], int size) {
    for (int i = 0; i < size; ++i) {
        if (abs(arr1[i] - arr2[i]) >= 1e-1) return false;
    }
    return true;
}


// CPU矩阵乘法: C[M][N] = A[M][K] * B[K][N]
void fp_gemm(const half* A, const half* B, float* C, 
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
                C[i * N + j] += static_cast<float>(A[i * K + k]) * 
                                static_cast<float>(B[k * N + j]);
            }
        }
    }
}

struct SM80_SPARSE_16x8x32_F32F16F16F32_TN{
    using DRegisters = float[4];
    using ARegisters = uint32_t[4];
    using BRegisters = uint32_t[4];
    using CRegisters = float[4];
    using ERegisters = uint32_t[1];

__device__ static void
  fma(float         & d0, float         & d1, float         & d2, float         & d3,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1, uint32_t const& b2, uint32_t const& b3,
      float    const& c0, float    const& c1, float    const& c2, float    const& c3,
    
      uint32_t const& e,int const id2){
            if (id2 == 0) {
                
            asm volatile(
                "mma.sp::ordered_metadata.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, "
                "{%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12,%13,%14,%15}, %16, 0x0;\n"
                : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                "r"(b2), "r"(b3), "f"(c0), "f"(c1), "f"(c2), "f"(c3),
                "r"(e));
                const int laneid = threadIdx.x % 32;
                    if(laneid == -1){
                    printf("laneid:%d\n",laneid);
                    printf("d_regs[0]=%.f\n",d0);
                    printf("d_regs[1]=%.f\n",d1);
                    printf("d_regs[2]=%.f\n",d2);
                    printf("d_regs[3]=%.f\n",d3);

                    printf("e:%d\n",e);

                    printf("a0:%d\n",a0);
                    printf("a1:%d\n",a1);
                    printf("a2:%d\n",a2);
                    printf("a3:%d\n",a3);
                    printf("b0:%d\n",b0);
                    printf("b1:%d\n",b1);
                    printf("b2:%d\n",b2);
                    printf("b3:%d\n",b3);
                    }
                
                    }
            else if (id2 == 1) {
            asm volatile(
                "mma.sp::ordered_metadata.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, "
                "{%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12,%13,%14,%15}, %16, 0x1;\n"
                : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                "r"(b2), "r"(b3), "f"(c0), "f"(c1), "f"(c2), "f"(c3),
                "r"(e));
            }
      }
    __device__ static void
    load_SP_A_half_G_row(const half *A ,ARegisters &a_regs){
        const int laneid = threadIdx.x % 32;
        const int group_row = laneid >> 2; // laneid // 4
        const int group_col = laneid % 4; // laneid % 4

        a_regs[0] = *(uint32_t *)&A[group_row * 16 + group_col * 2];
        a_regs[1] = *(uint32_t *)&A[(group_row + 8) * 16 + group_col * 2];
        a_regs[2] = *(uint32_t *)&A[group_row * 16 + group_col * 2 + 8];
        a_regs[3] = *(uint32_t *)&A[(group_row + 8) * 16 + group_col * 2 + 8];

        if(laneid == -1){
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

        printf("a0:%.1f\n",(float)A[group_row * 16 + group_col * 2]);
        printf("a1:%.1f\n",(float)A[group_row * 16 + group_col * 2 + 1]);


        printf("a2:%.1f\n",(float)A[(group_row + 8) * 16 + group_col * 2]);
        printf("a3:%.1f\n",(float)A[(group_row + 8) * 16 + group_col * 2 + 1]);


        printf("a4:%.1f\n",(float)A[group_row * 16 + group_col * 2 + 8]);
        printf("a5:%.1f\n",(float)A[group_row * 16 + group_col * 2 + 8 + 1]);

        printf("a6:%.1f\n",(float)A[(group_row + 8) * 16 + group_col * 2 + 8]);
        printf("a7:%.1f\n",(float)A[(group_row + 8) * 16 + group_col * 2 + 8 + 1]);
    }
    }
    __device__ static void
    load_SP_B_half_G_row(const half *B ,BRegisters &b_regs){
        const int laneid = threadIdx.x % 32;
        const int group_row = laneid % 4 * 2; 
        const int group_col = laneid >> 2; 
        half b0_7[8];
        b0_7[0] = B[group_row * 8 + group_col];
        b0_7[1] = B[(group_row + 1) * 8 + group_col];
        b0_7[2] = B[(group_row + 8) * 8 + group_col];
        b0_7[3] = B[(group_row + 8 + 1) * 8 + group_col];
        b0_7[4] = B[(group_row + 16) * 8 + group_col];
        b0_7[5] = B[(group_row + 16 + 1) * 8 + group_col];
        b0_7[6] = B[(group_row + 24) * 8 + group_col];
        b0_7[7] = B[(group_row + 24 + 1) * 8 + group_col];
        b_regs[0] = (*reinterpret_cast<uint16_t*>(&b0_7[1])<<16) | (*reinterpret_cast<uint16_t*>(&b0_7[0]));
        b_regs[1] = (*reinterpret_cast<uint16_t*>(&b0_7[3])<<16) | (*reinterpret_cast<uint16_t*>(&b0_7[2]));
        b_regs[2] = (*reinterpret_cast<uint16_t*>(&b0_7[5])<<16) | (*reinterpret_cast<uint16_t*>(&b0_7[4]));
        b_regs[3] = (*reinterpret_cast<uint16_t*>(&b0_7[7])<<16) | (*reinterpret_cast<uint16_t*>(&b0_7[6]));
        if(laneid == -1){
            printf("laneid:%d\n",laneid);
        printf("b0_7[0]%.f\n",(float)b0_7[0]);
        printf("b0_7[1]%.f\n",(float)b0_7[1]);
        printf("b0_7[2]%.f\n",(float)b0_7[2]);
        printf("b0_7[3]%.f\n",(float)b0_7[3]);

        printf("b0_7[4]%.f\n",(float)b0_7[4]);
        printf("b0_7[5]%.f\n",(float)b0_7[5]);
        printf("b0_7[6]%.f\n",(float)b0_7[6]);
        printf("b0_7[7]%.f\n",(float)b0_7[7]);
        }
    }

    __device__ static void
    load_SP_A_meta_data_u32_G_row(const uint32_t *meta_data ,ERegisters &e_regs){
        const int laneid = threadIdx.x % 32;
        e_regs[0] = meta_data[laneid];
        if(laneid == -1){
            printf("laneid:%d, e_regs[0]:%d\n",laneid,e_regs[0]);
        }
    }

    __device__ static void
    store_D(float *D,DRegisters &d_regs){
        const int laneid = threadIdx.x % 32;
        const int groupID = laneid >> 2; // laneid // 4
        const int threadID_in_group = (laneid % 4); 

        int row;
        int col;

        #pragma unroll
        for(int i=0;i<4;i++){
            if(i<2){
                row = groupID;
            }else{
                row = groupID + 8;
            }
            col = (threadID_in_group * 2) + (i & 0x1);
            D[row * 8 + col] = d_regs[i];
        
            if(laneid == -1){
                printf("laneid:%d\n",laneid);
                printf("laneid: %d,D[%d][%d]=%.f\n",laneid,row,col ,D[row * 8 + col]);
            }
        }
    }
};

__global__ void mma_SM80_SPARSE_16x8x32_F32F16F16F32_TN(half* input_A, half* input_B, float* input_C, float* output_D, uint32_t* meta_data,int id2){

    SM80_SPARSE_16x8x32_F32F16F16F32_TN::ARegisters a_regs;
    SM80_SPARSE_16x8x32_F32F16F16F32_TN::load_SP_A_half_G_row(input_A, a_regs);
    SM80_SPARSE_16x8x32_F32F16F16F32_TN::BRegisters b_regs;
    SM80_SPARSE_16x8x32_F32F16F16F32_TN::load_SP_B_half_G_row(input_B, b_regs);
    SM80_SPARSE_16x8x32_F32F16F16F32_TN::CRegisters c_regs = {0.0f, 0.0f, 0.0f, 0.0f};
    SM80_SPARSE_16x8x32_F32F16F16F32_TN::DRegisters d_regs = {0.0f, 0.0f, 0.0f, 0.0f};
    SM80_SPARSE_16x8x32_F32F16F16F32_TN::ERegisters e_regs;
    SM80_SPARSE_16x8x32_F32F16F16F32_TN::load_SP_A_meta_data_u32_G_row(meta_data, e_regs);
   
    SM80_SPARSE_16x8x32_F32F16F16F32_TN::fma(d_regs[0], d_regs[1], d_regs[2], d_regs[3],
        a_regs[0], a_regs[1], a_regs[2], a_regs[3],
        b_regs[0], b_regs[1], b_regs[2], b_regs[3],
        c_regs[0], c_regs[1], c_regs[2], c_regs[3],
        e_regs[0],id2);
    SM80_SPARSE_16x8x32_F32F16F16F32_TN::store_D(output_D, d_regs);
}

int main(){
    half    *host_a = new half[M * K];
    half    *host_b = new half[K * N];
    float   *host_c = new float[M * N];
    float   *host_d = new float[M * N];

    half    *host_sp_a = new half[M * K / 2];
    uint32_t *meta_data_a = new uint32_t[WARP_SIZE];

    half    *dev_sp_a;
    half    *dev_b;
    float   *dev_c;
    float   *dev_d;
    uint32_t *dev_meta_data_a;
    
    CHECK_CUDA(cudaMalloc(&dev_sp_a, sizeof(half)*M*K));
    CHECK_CUDA(cudaMalloc(&dev_b, sizeof(half)*K*N));
    CHECK_CUDA(cudaMalloc(&dev_c, sizeof(float)*K*N));
    CHECK_CUDA(cudaMalloc(&dev_d, sizeof(float)*M*N));
    CHECK_CUDA(cudaMalloc(&dev_meta_data_a, sizeof(uint32_t)*WARP_SIZE));
    // 初始化Matrix A
    for(int i = 0;i < M;i++){
        for(int j=0;j < K;j++){
            if(j % 4 == 0 || j % 4 == 2){
                host_a[i * K + j] = half(i+j+1);
            }else{
                host_a[i * K + j] = half(0);
            }
        }
    }
    for(int i = M/2;i < M;i++){
        for(int j=0;j < K;j++){
            if(j % 4 == 1 || j % 4 == 3){
                host_a[i * K + j] = half(i+j+1);
            }else{
                host_a[i * K + j] = half(0);
            }
        }
    }

    // 初始化meta_data_a,都是0和3号有数据，case比较简单
    for(int i = 0;i<WARP_SIZE;i++){
        meta_data_a[i] = (uint32_t)(0b11011101110111011000100010001000);
        // printf("meta_data_a[%d]:%d\n",i,meta_data_a[i]);
    }

    // 初始化Matrix B
    for(int i = 0; i < K*N; ++i)    host_b[i] =(i*1.0f); 
    //初始化sp Matrix A
    int index = 0;
    for(int i = 0;i < M * K;i++){
        if(host_a[i] != half(0)){
            host_sp_a[index++] = host_a[i];
        }
    }
    
    dumpEx_row_major(host_a, M, K, "Matrix A",8,4);
    dumpEx_row_major(host_sp_a, M,K / 2, "sp Matrix A",8,2);
    dumpEx_row_major(host_b, K,N, "Matrix B",8);

    CHECK_CUDA(cudaMemcpy(dev_sp_a, host_sp_a, sizeof(half)*M*K/2, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dev_b, host_b, sizeof(half)*K*N, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dev_c, host_c, sizeof(float)*K*N, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dev_d, host_d, sizeof(float)*M*N, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dev_meta_data_a, meta_data_a, sizeof(uint32_t)*WARP_SIZE, cudaMemcpyHostToDevice));

    dim3 grid(1);
    dim3 block(32);
    mma_SM80_SPARSE_16x8x32_F32F16F16F32_TN<<<grid,block>>>(dev_sp_a,dev_b,dev_c,dev_d,dev_meta_data_a,0);


    // 同步
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(host_d, dev_d, sizeof(float)*M*N, cudaMemcpyDeviceToHost));
    dumpEx_row_major(host_d, M,N, "gpu gemm",8);
    //cpu验证
    fp_gemm(host_a, host_b, host_c, M, N, K);
    dumpEx_row_major(host_c, M, N, "cpu gemm",8);

    compareArrays(host_c, host_d, M*N) ? 
        std::cout << "Results match!" << std::endl : 
        std::cout << "Results do not match!" << std::endl;
    return 0;
}