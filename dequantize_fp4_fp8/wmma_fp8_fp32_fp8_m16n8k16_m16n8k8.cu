#include "wmma.cuh"

#include <cstdint>
#include <cuda_fp4.h>
#include "wmma_utils.cuh"

#include "../config.h"

#define USE_SHARED_MEMORY 0

using fp4 = __nv_fp4_e2m1;

struct packed_float4_t{
    float data[4];
};

__device__ packed_float4_t ptx_m16n8k16_m16n8k8(fp8* input_A, float* input_B, fp8* input_C) {
    
    //---------------- 第一个tensor core计算开始------------
    const size_t laneid = threadIdx.x % WARP_SIZE;

    uint32_t RA[2]{0};
    uint32_t RB[1]{0};
    uint32_t RC[2]{0};
    uint32_t RD[2]{0};
    
    int index_a = laneid / 4 * 16 + laneid % 4 * 4;
    int index_b = laneid % 4 * N * 4 + laneid /4;
    RA[0] = *(uint32_t*)&input_A[index_a];
    RA[1] = *(uint32_t*)&input_A[index_a + K * 8];
    fp8 RB0 = static_cast<fp8>(input_B[index_b]);
    fp8 RB1 = static_cast<fp8>(input_B[index_b + 8]);
    fp8 RB2 = static_cast<fp8>(input_B[index_b + 16]);
    fp8 RB3 = static_cast<fp8>(input_B[index_b + 24]);
    RB[0] |= static_cast<uint32_t>(RB0.__x);
    RB[0] |= static_cast<uint32_t>(RB1.__x) << 8;
    RB[0] |= static_cast<uint32_t>(RB2.__x) << 16;
    RB[0] |= static_cast<uint32_t>(RB3.__x) << 24;


    //执行mma执行
    HMMA16816(RD[0], RD[1],
              RA[0], RA[1],
              RB[0],
              RC[0], RC[1]);   
    //---------------- 第一个tensor core计算结束------------
    //---------------- 第二个tensor core计算开始------------        
    SM75_16x8x8_F32F16F16F32_TN::BRegisters fragment_B;
    SM75_16x8x8_F32F16F16F32_TN::load_B_fp8_G(input_C, fragment_B);
    // initialize D
    SM75_16x8x8_F32F16F16F32_TN::DRegisters fragment_D = {0.0f, 0.0f, 0.0f, 0.0f};
    // execute the MMA operation
    SM75_16x8x8_F32F16F16F32_TN::fma(fragment_D[0], fragment_D[1], fragment_D[2], fragment_D[3],
        RD[0], RD[1],
        fragment_B[0],
        fragment_D[0], fragment_D[1], fragment_D[2], fragment_D[3]);
    packed_float4_t res;
    res.data[0] = fragment_D[0];
    res.data[1] = fragment_D[0];
    res.data[2] = fragment_D[0];
    res.data[3] = fragment_D[0];
    return res;

}



// D=A×B×C
__global__ void ptx_m16n8k16_m16n8k8_kernel(fp8* input_A, float* input_B, fp8* input_C, float* output_D) {
    
    //---------------- 第一个tensor core计算开始------------
    const size_t laneid = threadIdx.x % WARP_SIZE;

    uint32_t RA[2]{0};
    uint32_t RB[1]{0};
    uint32_t RC[2]{0};
    uint32_t RD[2]{0};
    
    int index_a = laneid / 4 * 16 + laneid % 4 * 4;
    int index_b = laneid % 4 * N * 4 + laneid /4;

    RA[0] = *(uint32_t*)&input_A[index_a];
    RA[1] = *(uint32_t*)&input_A[index_a + K * 8];
    fp8 RB0 = static_cast<fp8>(input_B[index_b]);
    fp8 RB1 = static_cast<fp8>(input_B[index_b + 8]);
    fp8 RB2 = static_cast<fp8>(input_B[index_b + 16]);
    fp8 RB3 = static_cast<fp8>(input_B[index_b + 24]);
    RB[0] |= static_cast<uint32_t>(RB0.__x);
    RB[0] |= static_cast<uint32_t>(RB1.__x) << 8;
    RB[0] |= static_cast<uint32_t>(RB2.__x) << 16;
    RB[0] |= static_cast<uint32_t>(RB3.__x) << 24;
    


    //执行mma执行
    HMMA16816(RD[0], RD[1],
              RA[0], RA[1],
              RB[0],
              RC[0], RC[1]); 
    //---------------- 第一个tensor core计算结束------------
    //---------------- 第二个tensor core计算开始------------        
    SM75_16x8x8_F32F16F16F32_TN::ARegisters fragment_A;
    // load A
    fragment_A[0] = RD[0];
    fragment_A[1] = RD[1];
    // load B
    SM75_16x8x8_F32F16F16F32_TN::BRegisters fragment_B;
    SM75_16x8x8_F32F16F16F32_TN::load_B_fp8_G(input_C, fragment_B);
    // initialize C
    SM75_16x8x8_F32F16F16F32_TN::CRegisters fragment_C = {0.0f, 0.0f, 0.0f, 0.0f};
    // initialize D
    SM75_16x8x8_F32F16F16F32_TN::DRegisters fragment_D = {0.0f, 0.0f, 0.0f, 0.0f};
    // execute the MMA operation
    SM75_16x8x8_F32F16F16F32_TN::fma(fragment_D[0], fragment_D[1], fragment_D[2], fragment_D[3],
        RD[0], RD[1],
        fragment_B[0],
        fragment_C[0], fragment_C[1], fragment_C[2], fragment_C[3]);

    SM75_16x8x8_F32F16F16F32_TN::store_D(output_D, fragment_D);
}


int main(){
    
    fp8     *host_a = new fp8[M*K];
    float   *host_b = new float[K*N];
    fp8     *host_c = new fp8[K1*N1];
    float   *host_d = new float[M*N];

    fp8     *dev_a;
    float   *dev_b;
    fp8     *dev_c;
    float   *dev_d;

    CHECK_CUDA(cudaMalloc(&dev_a, sizeof(fp8)*M*K));
    CHECK_CUDA(cudaMalloc(&dev_b, sizeof(float)*K*N));
    CHECK_CUDA(cudaMalloc(&dev_c, sizeof(fp8)*K1*N1));
    CHECK_CUDA(cudaMalloc(&dev_d, sizeof(float)*M*N));
    // 主机数据初始化
    // A矩阵为对角矩阵
    for(int i=0;i<M;i++){
        for(int j=0;j<K;j++){
            if(i == j)
                host_a[i*K+j] = static_cast<fp8>(1.0f);
            else
                host_a[i*K+j] = static_cast<fp8>(0.0f);
        }
    }
    // B矩阵行主序递增
    for(int i = 0; i < K*N; ++i)    host_b[i] = (i*1.0f); 
    for(int i = 0; i < 16 * 8;i++){

                if(i%8 == 0)                host_b[i] = 288.0f;
                if(i%8 == 1)                host_b[i] = -192.0f;
                if(i%8 == 2)                host_b[i] = -144.0f;
                if(i%8 == 3)                host_b[i] = -96.0f;
                if(i%8 == 4)                host_b[i] = -72.0f;
                if(i%8 == 5)                host_b[i] = -48.0f;
                if(i%8 == 6)                host_b[i] = -24.0f;
                if(i%8 == 7)                host_b[i] = 0.0f;
        
    }

    // C矩阵为单位对角矩阵
    for(int i=0;i<K1;i++){
        for(int j=0;j<N1;j++){
            if(i == j)
                host_c[i*N1+j] = static_cast<fp8>(1.0f);
            else
                host_c[i*N1+j] = static_cast<fp8>(0.0f);
        }
    }
    for(int i = 0; i < M*N; ++i)    host_d[i] = 0.0f;
    // 打印一下
    #ifndef TEST_TIME
    printf("host_a:\n");
    dumpEx1<fp8>(host_a, M, K,"host_a");
    printf("host_b:\n");
    dumpEx1<float>(host_b, K, N,"host_b");
    printf("host_c:\n");
    dumpEx1<fp8>(host_c, K1, N1,"host_c");
    #endif // TEST_TIME

    CHECK_CUDA(cudaMemcpy(dev_a, host_a, sizeof(fp8)*M*K, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dev_b, host_b, sizeof(float)*K*N, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dev_c, host_c, sizeof(fp8)*K1*N1, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dev_d, host_d, sizeof(float)*M*N, cudaMemcpyHostToDevice));

    dim3 block(32);
    dim3 grid(1);

    #ifdef TEST_TIME
    // 创建CUDA事件计时
    // warm up
    for(int i = 0; i < ITERS ; i++) {
        // Warm up
        ptx_m16n8k16_m16n8k8_kernel<<<grid, block>>>(dev_a, dev_b, dev_c, dev_d);
    }
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for(int i = 0; i < ITERS ; i++) {
    #endif
    ptx_m16n8k16_m16n8k8_kernel<<<grid, block>>>(dev_a, dev_b, dev_c, dev_d);
    #ifdef TEST_TIME
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    printf("单次反量化运行时间: %.12f ms\n", (float)(ms/ITERS));//
    #endif
    // 拷贝结果回主机
    cudaDeviceSynchronize();

    CHECK_CUDA(cudaMemcpy(host_d, dev_d, sizeof(float)*M*N, cudaMemcpyDeviceToHost));
    #ifndef TEST_TIME
    printf("output_d:\n");
    dumpEx1<float>(host_d, M, N,"output_d");
    #endif // TEST_TIME

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