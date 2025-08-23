// tee mma_ops.cu<<-'EOF'
#include <iostream>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cuda.h>
#include <mma.h>

using namespace nvcuda;

#define WARP_SIZE 32

#define CHECK_CUDA(status)                                              \
  {                                                                     \
    cudaError_t error = status;                                         \
    if (error != cudaSuccess) {                                         \
      std::cerr << "Got bad cuda status: " << cudaGetErrorString(error) \
                << " at line: " << __LINE__ << std::endl;               \
      exit(EXIT_FAILURE);                                               \
    }                                                                   \
  }

//mma指令
#define HMMA16816(RD0, RD1, RA0, RA1, RA2, RA3, RB0, RB1, RC0, RC1)                                                    \
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};\n" \
                 : "=r"(RD0), "=r"(RD1)                                                                                \
                 : "r"(RA0), "r"(RA1), "r"(RA2), "r"(RA3), "r"(RB0), "r"(RB1), "r"(RC0), "r"(RC1))

//加载A矩阵(行存储)
#define LDMATRIX_X4(R0, R1, R2, R3, addr)                                             \
    asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n" \
                 : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3)                             \
                 : "l"(addr))

#define LDMATRIX_X4_TRANS(R0, R1, R2, R3, addr)                                             \
    asm volatile("ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n" \
                 : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3)                             \
                 : "l"(addr))

//加载B矩阵(行存储),需要转置
#define LDMATRIX_X2(R0, R1, addr) \
    asm volatile("ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0, %1}, [%2];\n" : "=r"(R0), "=r"(R1) : "l"(addr))

//异步加载数据
#define CP_ASYNC_CG(dst, src, Bytes) \
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(Bytes))
#define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;\n" ::)
#define CP_ASYNC_WAIT_GROUP(N) asm volatile("cp.async.wait_group %0;\n" ::"n"(N))
#define CP_ASYNC_WAIT_ALL() asm volatile("cp.async.wait_all;\n" ::)

__global__ void ptx_m16n8k16_kernel(half* input_A, half* input_B, half* input_C, int M, int N, int K) {
    
    const size_t laneid = threadIdx.x % WARP_SIZE;
    
    __shared__ half A[16*16];
    __shared__ half B[16*16];
    //为了保证对比的公平,都先加载到share memory里
    uint32_t a_smem_lane_addr = __cvta_generic_to_shared(&A[laneid*8]); 
    CP_ASYNC_CG(a_smem_lane_addr,&input_A[laneid*8],16);
     
    uint32_t b_smem_lane_addr = __cvta_generic_to_shared(&B[laneid*8]); 
    CP_ASYNC_CG(b_smem_lane_addr,&input_B[laneid*8],16); 

    CP_ASYNC_COMMIT_GROUP();
    CP_ASYNC_WAIT_GROUP(0);
    __syncthreads();
    
    clock_t begin=clock64();
    uint32_t RA[4];
    uint32_t RB[2][2];
    uint32_t RC[2][2];
    
    RC[0][0]=0;
    RC[0][1]=0;
    
    RC[1][0]=0;
    RC[1][1]=0;
    
    int aTile_index = laneid % 16 * 16 + laneid / 16 * 8;
    LDMATRIX_X4(RA[0], RA[1], RA[2], RA[3], __cvta_generic_to_shared(&A[aTile_index]));
    LDMATRIX_X2(RB[0][0], RB[0][1],__cvta_generic_to_shared(&B[laneid % 16 * 16]));
    LDMATRIX_X2(RB[1][0],RB[1][1],__cvta_generic_to_shared(&B[laneid % 16 * 16+8]));
    
    //执行mma执行
    HMMA16816(RC[0][0], RC[0][1],
              RA[0], RA[1], RA[2], RA[3],
              RB[0][0], RB[0][1],
              RC[0][0], RC[0][1]);   
    HMMA16816(RC[1][0], RC[1][1],
              RA[0], RA[1], RA[2], RA[3],
              RB[1][0], RB[1][1],
              RC[1][0], RC[1][1]);
              
    //C矩阵 M*N=16*8
    /*
    groupID           = %laneid >> 2
    threadID_in_group = %laneid % 4

    row =    groupID                                 for ci where i <  2
             groupID + 8                             for ci where i >= 2

    col =  (threadID_in_group * 2) + (i & 0x1)       for ci where i = {0,..,3}
    */

    int groupID           = laneid /4;
    int threadID_in_group = laneid % 4;
    
    int row_c0 = groupID;
    int col_c0 = (threadID_in_group * 2) + (0 & 0x1);
    
    int row_c2 = groupID + 8;
    int col_c2 = (threadID_in_group * 2) + (2 & 0x1);              
              
    //写回到DRAM
    *(uint32_t*)&input_C[row_c0*N+col_c0]=RC[0][0];
    *(uint32_t*)&input_C[row_c2*N+col_c2]=RC[0][1];    
    *(uint32_t*)&input_C[row_c0*N+8+col_c0]=RC[1][0];
    *(uint32_t*)&input_C[row_c2*N+8+col_c2]=RC[1][1];    

    clock_t end=clock64();
    
    if(laneid==0)
    {
        printf("ptx_mma_shared kernel e2e(cycles):%ld\n",end-begin);
    }    
}

__global__ void wmma_api_kernel(half *dev_a, half *dev_b,half *dev_c) {
    int tid  = threadIdx.x + blockIdx.x * blockDim.x;

    __shared__ half A[16*16];
    __shared__ half B[16*16];
    
    //为了保证对比的公平,都先加载到share memory里
    uint32_t a_smem_lane_addr = __cvta_generic_to_shared(&A[threadIdx.x*8]); 
    CP_ASYNC_CG(a_smem_lane_addr,&dev_a[threadIdx.x*8],16);
     
    uint32_t b_smem_lane_addr = __cvta_generic_to_shared(&B[threadIdx.x*8]); 
    CP_ASYNC_CG(b_smem_lane_addr,&dev_b[threadIdx.x*8],16); 

    CP_ASYNC_COMMIT_GROUP();
    CP_ASYNC_WAIT_GROUP(0);
    __syncthreads();
    
    clock_t begin=clock64();
    nvcuda::wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    nvcuda::wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    nvcuda::wmma::fragment<wmma::accumulator, 16, 16, 16, half> c_frag;    
    wmma::load_matrix_sync(a_frag, A, 16);
    wmma::load_matrix_sync(b_frag, B, 16);
    wmma::fill_fragment(c_frag, 0.0f);
    nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag); 
    wmma::store_matrix_sync(dev_c, c_frag, 16, wmma::mem_row_major);
    clock_t end=clock64();
    if(tid==0)
    {
        printf("wmma_kernel e2e(cycles):%ld\n",end-begin);
    } 
}





int M=16;
int N=16;    
int K=16;

void dump(half *host_c)
{
    for(int r=0;r<M;r++)
    {
       for(int c=0;c<N;c++)
       {
        printf("%8.3f ",__half2float(host_c[r*N+c]));
       }
       printf("\n");
    }
}

// 一维数组矩阵转置函数 (支持任意尺寸矩阵)
void transpose1DMatrix(half* input, half* output, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            // 核心转置操作：行列索引互换
            output[j * rows + i] = input[i * cols + j]; // [2,7](@ref)
        }
    }
}

int main() {
    half *host_a = new half[M*K];
    half *host_b = new half[K*N];
    half *host_c = new half[M*N];
    half *host_b_trans = new half[M*N];
    half *dev_a;
    half *dev_b;
    half *dev_c;
    
    CHECK_CUDA(cudaMalloc(&dev_a, sizeof(half)*M*K));
    CHECK_CUDA(cudaMalloc(&dev_b, sizeof(half)*K*N));
    CHECK_CUDA(cudaMalloc(&dev_c, sizeof(half)*M*N));
    
    for(int i = 0; i < M*K; ++i) host_a[i] = __float2half(i*0.01);
    for(int i = 0; i < K*N; ++i) host_b[i] = __float2half(i*0.01);
    
    transpose1DMatrix(host_b, host_b_trans, 16, 16);

    for(int j=0;j<1;j++)
    {
        CHECK_CUDA(cudaMemcpy(dev_a, host_a, sizeof(half)*M*K,cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(dev_b, host_b, sizeof(half)*K*N,cudaMemcpyHostToDevice));
        for(int i = 0; i < M*N; ++i) host_c[i] = 0;
        CHECK_CUDA(cudaMemcpy(dev_c, host_c, sizeof(half)*K*N,cudaMemcpyHostToDevice));
      
        ptx_m16n8k16_kernel<<<1, 32>>>(dev_a, dev_b,dev_c,M,N,K);cudaDeviceSynchronize();
        cudaMemcpy(host_c, dev_c, sizeof(half)*M*N, cudaMemcpyDeviceToHost);
        dump(host_c);
        
        printf("-------------------------------------------------------------\n");
        for(int i = 0; i < M*N; ++i) host_c[i] = 0;
        CHECK_CUDA(cudaMemcpy(dev_c, host_c, sizeof(half)*K*N,cudaMemcpyHostToDevice));  

        CHECK_CUDA(cudaMemcpy(dev_b, host_b_trans, sizeof(half)*K*N,cudaMemcpyHostToDevice));
        
        wmma_api_kernel<<<1, 32>>>(dev_a, dev_b,dev_c);cudaDeviceSynchronize();

        cudaMemcpy(host_c, dev_c, sizeof(half)*M*N, cudaMemcpyDeviceToHost);
        dump(host_c);
    }
    
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    free(host_a);
    free(host_b);
    free(host_c);
    return 0;
}
// EOF

// /usr/local/cuda/bin/nvcc -std=c++17 -O2 -arch=sm_86 -lineinfo mma_ops.cu -o mma_ops
// ./mma_ops

// /usr/local/NVIDIA-Nsight-Compute/ncu --set full --section SpeedOfLight_HierarchicalTensorRooflineChart --target-processes all --clock-control=none \
//                 --print-details all --export ncu_report_mma_diff_ops -f ./mma_ops

// # 查看tensor core利用率
// /usr/local/NVIDIA-Nsight-Compute/ncu --metrics \
// sm__ops_path_tensor_src_fp16_dst_fp16_sparsity_off.sum.pct_of_peak_sustained_elapsed,\
// sm__ops_path_tensor_src_fp16_dst_fp16_sparsity_off.sum,\
// sm__ops_path_tensor_src_fp16_dst_fp16_sparsity_off.sum.peak_sustained,\
// sm__ops_path_tensor_src_fp16_dst_fp16_sparsity_off.sum.per_second,\
// sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,\
// sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,\
// sm__cycles_elapsed ./mma_ops
