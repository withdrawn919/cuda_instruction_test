#include "../dequantize_fp4_fp8/wmma_utils.cuh"
#include "../dequantize_fp4_fp8/wmma.cuh"

#include <cstdint>
#include <iostream>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cuda.h>
#include <mma.h>

#include <inttypes.h>


// #define M8N8_X1
// #define M8N8_X2
// #define M8N8_X4

// #define M8N8_TRANS_X1
// #define M8N8_TRANS_X2
#define M8N8_TRANS_X4

#define LDMATRIX_X1(R0, addr)                                             \
    asm volatile("ldmatrix.sync.aligned.x1.m8n8.shared.b16 {%0}, [%1];\n" \
                 : "=r"(R0)                                               \
                 : "l"(addr))

#define LDMATRIX_TRANS_X1(R0, addr)                                             \
    asm volatile("ldmatrix.sync.aligned.x1.trans.m8n8.shared.b16 {%0}, [%1];\n" \
                 : "=r"(R0)                                               \
                 : "l"(addr))

#define LDMATRIX_X2(R0, R1, addr)                                             \
    asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n" \
                 : "=r"(R0), "=r"(R1)                                         \
                 : "l"(addr))
            
#define LDMATRIX_TRANS_X2(R0, R1, addr)                                             \
    asm volatile("ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0, %1}, [%2];\n" \
                 : "=r"(R0), "=r"(R1)                                         \
                 : "l"(addr))

#define LDMATRIX_X4(R0, R1, R2, R3, addr)                                             \
    asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n" \
                 : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3)                             \
                 : "l"(addr))

#define LDMATRIX_TRANS_X4(R0, R1, R2, R3, addr)                                             \
    asm volatile("ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n" \
                 : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3)                             \
                 : "l"(addr))
// m8n16的ldmatrix需要地址对齐到16B
#define LDMATRIX_m8n16_X1(R0, addr)                                                             \
    asm volatile("ldmatrix.sync.aligned.m8n16.x1.shared.b8x16.b4x16_p64 {%0}, [%1];\n"          \
                 : "=r"(R0)                                                                     \
                 : "l"(addr))

__global__ void ldmatrix_m8n16_4bit_x1(){
    __shared__ __align__(4) uint8_t A[8 * 8 * 2];//每一个8bit存两个4bit的数
    
    const int laneid = threadIdx.x % 32;
    A[laneid] = 0b10100101;
    A[laneid + 32] = 0b00001111;
    A[laneid + 32 + 32] = 0b00000000;
    A[laneid + 32 + 32 + 32] = 0b00000000;

    // A[laneid + 32 + 32 + 32 + 32] = 0b00111100;
    // A[laneid + 32 + 32 + 32 + 32 + 32] = 0b00000000;
    // A[laneid + 32 + 32 + 32 + 32 + 32 +32] = 0b00111100;
    // A[laneid + 32 + 32 + 32 + 32 + 32 +32 + 32] = 0b00000000;
    __syncthreads();
    if(laneid == -1){
        dumpEx_row_major<uint8_t>(A, 16,8, "8bit A:",1,4);
    }
    uint32_t RA;

    
    int aTile_index = laneid * 16 % (8 * 8);
    uint32_t smem_int_ptr = __cvta_generic_to_shared(&A[aTile_index]);
    // printf("laneid = %d, aTile_index = %d, smem_int_ptr = 0x%x\n",laneid,aTile_index,smem_int_ptr);
        asm volatile ("ldmatrix.sync.aligned.m8n16.x1.shared.b8x16.b4x16_p64  {%0}, [%1];\n"//b4x16_p64
        : "=r"(RA)
        :  "r"(smem_int_ptr));

    if(laneid == -1){
        printf("laneid = %d\n",static_cast<int>(laneid));
        print_binary32(RA);
    }
    
}
// ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
//                                 ldmatrix.X1                       
// ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
__global__ void ldmatrix_m8n8_x1(){
    __shared__ half A[8*8];
    const size_t laneid = threadIdx.x % 32;
    A[laneid]=__float2half(laneid * 1.0f);
    A[laneid + 32]=__float2half(laneid * 2.0f);
    __syncthreads();

    if(laneid == 0){
        dumpEx_row_major<half>(A, 8,8, "16bit A:",1,2);
    }

    uint32_t RA;

    LDMATRIX_X1(RA, __cvta_generic_to_shared(&A[(laneid * 8) % (8 * 8)]));

    int x = static_cast<int>(RA);

    // 1. 将 uint32_t 拆分为两个 uint16_t
    uint16_t high16_x = static_cast<uint16_t>(x >> 16);  // 高16位
    uint16_t low16_x  = static_cast<uint16_t>(x & 0xFFFF); // 低16位

    // 2. 将 uint16_t 的位模式重新解释为 half 类型
    __half h_high_x, h_low_x;

    // 使用 memcpy 或类型双关来保持位模式不变
    memcpy(&h_high_x, &high16_x, sizeof(uint16_t));
    memcpy(&h_low_x, &low16_x, sizeof(uint16_t));



        // 只让一个线程（例如laneid=0）执行打印，避免输出混乱
       if (laneid == 0) {
        printf("==============================================PTX LDMATRX=====================================\n");
        printf("| LaneId | Variable     | Raw Hex  | Float Value  | | Variable     | Raw Hex  | Float Value  |\n");
        printf("|--------|--------------|----------|--------------| |--------------|----------|--------------|\n");
        }

        // 所有线程同步，确保表头先打印（如果多个线程都打印表头会混乱，所以只让一个线程打印表头）
        __syncthreads(); // 如果这是在block级别，需要同步

        // 每个线程打印自己的信息，但将X的高16位和低16位打印在同一行
        printf("| %6d | X (Low 16)  |   0x%04X | %12.6f | "
            "| X (High 16)   |   0x%04X | %12.6f |\n", 
            static_cast<int>(laneid), 
            low16_x, __half2float(h_low_x),
            high16_x, __half2float(h_high_x));


}
// cuda替换ptx-ldmatirx.m8n8.x1
__global__ void load_m8n8_x1(){
    __shared__ half A[8*8];
    const int laneid = (threadIdx.z * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x) 
                        & (warpSize - 1);
    A[laneid]=__float2half(laneid * 1.0f);
    A[laneid + 32]=__float2half(laneid * 2.0f);
    __syncthreads();

    if(laneid == 0){
        dumpEx_row_major<half>(A, 8,8, "16bit A:",1,2);
    }

    uint32_t *ptr = reinterpret_cast<uint32_t*>(&A[(laneid * 8) % (8 * 8)]);

    uint32_t *cur_ptr;
    #pragma unroll 8// 指示编译器将后续循环展开8次
    for(int i=0;i<8;i++){
        uintptr_t temp;
        temp = __shfl_sync(0xFFFFFFFF, reinterpret_cast<uintptr_t>(ptr), i);
        if(laneid >= i * 4 && laneid < (i + 1) * 4){
            cur_ptr = reinterpret_cast<uint32_t*>(temp) + (laneid % 4);
        }
    }

    int x = *cur_ptr;
    if (1){
        // 1. 将 uint32_t 拆分为两个 uint16_t
        uint16_t high16_x = static_cast<uint16_t>(x >> 16);  // 高16位
        uint16_t low16_x  = static_cast<uint16_t>(x & 0xFFFF); // 低16位

        // 2. 将 uint16_t 的位模式重新解释为 half 类型
        __half h_high_x, h_low_x;

        // 使用 memcpy 或类型双关来保持位模式不变
        memcpy(&h_high_x, &high16_x, sizeof(uint16_t));
        memcpy(&h_low_x, &low16_x, sizeof(uint16_t));



        // 只让一个线程（例如laneid=0）执行打印，避免输出混乱
       if (laneid == 0) {
        printf("==============================================PTX LDMATRX=====================================\n");
        printf("| LaneId | Variable     | Raw Hex  | Float Value  | | Variable     | Raw Hex  | Float Value  |\n");
        printf("|--------|--------------|----------|--------------| |--------------|----------|--------------|\n");
        }

        // 所有线程同步，确保表头先打印（如果多个线程都打印表头会混乱，所以只让一个线程打印表头）
        __syncthreads(); // 如果这是在block级别，需要同步

        // 每个线程打印自己的信息，但将X的高16位和低16位打印在同一行
        printf("| %6d | X (Low 16)  |   0x%04X | %12.6f | "
            "| X (High 16)   |   0x%04X | %12.6f |\n", 
            static_cast<int>(laneid), 
            low16_x, __half2float(h_low_x),
            high16_x, __half2float(h_high_x));

    }
}
// ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
//                            ldmatrix.trans.X1                       
// ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
__global__ void ldmatrix_m8n8_trans_x1(){
    constexpr int row = 16;
    constexpr int col = 8;
    constexpr int x_align = 1;
    __shared__ half A[row*col];
    const size_t laneid = threadIdx.x % 32;
    A[laneid]=__float2half(laneid * 1.0f);
    A[laneid + 32]=__float2half(laneid * 2.0f);
    A[laneid + 32 + 32]=__float2half(laneid * 3.0f);
    A[laneid + 32 + 32 + 32]=__float2half(laneid * 4.0f);

    __syncthreads();

    if(laneid == 0){
        dumpEx_row_major<half>(A, row,col, "16bit A:",1,2);
    }

    uint32_t RA;

    int index_temp = x_align * (laneid * 8) % (row * col);

    printf("index_temp: %d\n", index_temp);

    LDMATRIX_TRANS_X1(RA, __cvta_generic_to_shared(&A[x_align * ((laneid * 8) % (row * col))]));
    
    int x = static_cast<int>(RA);

    // 1. 将 uint32_t 拆分为两个 uint16_t
    uint16_t high16_x = static_cast<uint16_t>(x >> 16);  // 高16位
    uint16_t low16_x  = static_cast<uint16_t>(x & 0xFFFF); // 低16位

    // 2. 将 uint16_t 的位模式重新解释为 half 类型
    __half h_high_x, h_low_x;

    // 使用 memcpy 或类型双关来保持位模式不变
    memcpy(&h_high_x, &high16_x, sizeof(uint16_t));
    memcpy(&h_low_x, &low16_x, sizeof(uint16_t));


if (laneid == 0) {
    printf("==============================================PTX LDMATRX=====================================\n");
    printf("| LaneId | Variable     | Raw Hex  | Float Value  | | Variable     | Raw Hex  | Float Value  |\n");
    printf("|--------|--------------|----------|--------------| |--------------|----------|--------------|\n");
}

// 所有线程同步，确保表头先打印
__syncthreads();

// 每个线程打印自己的信息，但将X的高16位和低16位打印在同一行
printf("| %6d | X (Low 16)  |   0x%04X | %12.6f | "
       "| X (High 16)   |   0x%04X | %12.6f |\n", 
       static_cast<int>(laneid), 
       low16_x, __half2float(h_low_x),
       high16_x, __half2float(h_high_x));

}

// cuda替换ptx-ldmatirx.m8n8.trans.x1
__global__ void load_m8n8_trans_x1(){
    __shared__ half A[8*8];
    const int laneid = (threadIdx.z * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x) 
                        & (warpSize - 1);
    A[laneid]=__float2half(laneid * 1.0f);
    A[laneid + 32]=__float2half(laneid * 2.0f);
    __syncthreads();

    if(laneid == 0){
        dumpEx_row_major<half>(A, 8,8, "16bit A:",1,2);
    }

    half *ptr = reinterpret_cast<half*>(&A[(laneid * 8) % (8 * 8)]);


    __half2 h2_x = {0,0};

    uintptr_t temp = 0;

    #pragma unroll 4
    for(int i = 0; i < 4; i++) {
        temp = __shfl_sync(0xFFFFFFFF, reinterpret_cast<uintptr_t>(ptr), 2 * i);
        if(laneid % 4 == i){
            h2_x.x = *(reinterpret_cast<half*>(temp) + (laneid / 4));
        }
        temp = __shfl_sync(0xFFFFFFFF, reinterpret_cast<uintptr_t>(ptr), 2 * i + 1);
        if(laneid % 4 == i){
            h2_x.y = *(reinterpret_cast<half*>(temp) + (laneid / 4));            
        }
    }
    int x = *reinterpret_cast<int*>(&h2_x);



    if (1){


        // 1. 将 uint32_t 拆分为两个 uint16_t
        uint16_t high16_x = static_cast<uint16_t>(x >> 16);  // 高16位
        uint16_t low16_x  = static_cast<uint16_t>(x & 0xFFFF); // 低16位

        // 2. 将 uint16_t 的位模式重新解释为 half 类型
        __half h_high_x, h_low_x;

        // 使用 memcpy 或类型双关来保持位模式不变
        memcpy(&h_high_x, &high16_x, sizeof(uint16_t));
        memcpy(&h_low_x, &low16_x, sizeof(uint16_t));



        if (laneid == 0) {
            printf("===============================================SOFT LOAD======================================\n");
            printf("| LaneId | Variable     | Raw Hex  | Float Value  | | Variable     | Raw Hex  | Float Value  |\n");
            printf("|--------|--------------|----------|--------------| |--------------|----------|--------------|\n");
        }

        // 所有线程同步，确保表头先打印
        __syncthreads();

        // 每个线程打印自己的信息，但将X的高16位和低16位打印在同一行
        printf("| %6d | X (Low 16)  |   0x%04X | %12.6f | "
            "| X (High 16)   |   0x%04X | %12.6f |\n", 
            static_cast<int>(laneid), 
            (uint32_t)h_high_x, __half2float(h_low_x),
            (uint32_t)h_low_x, __half2float(h_high_x));

    }
}
// ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
//                                 ldmatrix.X2                       
// ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
__global__ void ldmatrix_m8n8_x2(){
    __shared__ half A[16*8];
    const size_t laneid = threadIdx.x % 32;
    A[laneid]=__float2half(laneid * 1.0f);
    A[laneid + 32]=__float2half(laneid * 2.0f);
    A[laneid + 32 + 32]=__float2half(laneid * 3.0f);
    A[laneid + 32 + 32 + 32]=__float2half(laneid * 4.0f);
    __syncthreads();

    uint32_t RA[2];

    LDMATRIX_X2(RA[0], RA[1], __cvta_generic_to_shared(&A[(laneid * 8) % (16 * 8)]));
}
// cuda替换ptx-ldmatirx.m8n8.x2
__global__ void load_m8n8_x2(){
    __shared__ half A[16*8];
    const int laneid = (threadIdx.z * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x) 
                        & (warpSize - 1);
    A[laneid]=__float2half(laneid * 1.0f);
    A[laneid + 32]=__float2half(laneid * 2.0f);
    A[laneid + 32 + 32]=__float2half(laneid * 3.0f);
    A[laneid + 32 + 32 + 32]=__float2half(laneid * 4.0f);
    __syncthreads();

    if(laneid == 0){
        dumpEx_row_major<half>(A, 16,8, "16bit A:",1,2);
    }

    uint32_t *ptr = reinterpret_cast<uint32_t*>(&A[(laneid * 8) % (16 * 8)]);

    uint32_t *cur_ptr_x = nullptr;
    uint32_t *cur_ptr_y = nullptr;
    #pragma unroll 16// 指示编译器将后续循环展开16次
    for(int i=0;i<16;i++){
        uintptr_t temp;
        temp = __shfl_sync(0xFFFFFFFF, reinterpret_cast<uintptr_t>(ptr), i);
        if(laneid >= i * 4&& laneid < (i + 1) * 4){
            cur_ptr_x = reinterpret_cast<uint32_t*>(temp) + (laneid % 4);

        }else if(laneid >= (i - 8) * 4 && laneid < (i + 1 - 8) * 4){
            cur_ptr_y = reinterpret_cast<uint32_t*>(temp) + (laneid % 4);
        }
    }

    // printf("Device: cur_ptr_x = %p  value = %f  LaneId=%d\n", cur_ptr_x, __half2float(*(half*)cur_ptr_x), laneid);
    // printf("Device: cur_ptr_y = %p  value = %f  LaneId=%d\n", cur_ptr_y, __half2float(*(half*)cur_ptr_y), laneid);

    int x = *cur_ptr_x;
    int y = *cur_ptr_y;

    // 1. 将 uint32_t 拆分为两个 uint16_t
    uint16_t high16_x = static_cast<uint16_t>(x >> 16);  // 高16位
    uint16_t low16_x  = static_cast<uint16_t>(x & 0xFFFF); // 低16位
    uint16_t high16_y = static_cast<uint16_t>(y >> 16);  // 高16位
    uint16_t low16_y  = static_cast<uint16_t>(y & 0xFFFF); // 低16位
    // 2. 将 uint16_t 的位模式重新解释为 half 类型
    __half h_high_x, h_low_x;
    __half h_high_y, h_low_y;
    // 使用 memcpy 或类型双关来保持位模式不变
    memcpy(&h_high_x, &high16_x, sizeof(uint16_t));
    memcpy(&h_low_x, &low16_x, sizeof(uint16_t));
    memcpy(&h_high_y, &high16_y, sizeof(uint16_t));
    memcpy(&h_low_y, &low16_y, sizeof(uint16_t));


    if (laneid == 0) {
        printf("==============================================PTX LDMATRX=====================================\n");
        printf("| LaneId | Variable     | Raw Hex  | Float Value  | | Variable     | Raw Hex  | Float Value  |\n");
        printf("|--------|--------------|----------|--------------| |--------------|----------|--------------|\n");
    }

    // 所有线程同步，确保表头先打印（如果多个线程都打印表头会混乱，所以只让一个线程打印表头）
    __syncthreads(); // 如果这是在block级别，需要同步

    // 每个线程打印自己的信息，但将X的高16位和低16位打印在同一行
    printf("| %6d | X (Low 16)  |   0x%04X | %12.6f | "
        "| X (High 16)   |   0x%04X | %12.6f |\n", 
        static_cast<int>(laneid), 
        low16_x, __half2float(h_low_x),
        high16_x, __half2float(h_high_x));
    printf("| %6d | Y (Low 16)  |   0x%04X | %12.6f | "
        "| Y (High 16)   |   0x%04X | %12.6f |\n", 
        static_cast<int>(laneid), 
        low16_y, __half2float(h_low_y),
        high16_y, __half2float(h_high_y));
}
// ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
//                            ldmatrix.trans.X2                       
// ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
__global__ void ldmatrix_m8n8_trans_x2(){
    constexpr int row = 16;
    constexpr int col = 8;
    __shared__ half A[row * col];
    const int laneid = (threadIdx.z * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x) 
                        & (warpSize - 1);
    A[laneid]=__float2half(laneid * 1.0f);
    A[laneid + 32]=__float2half(laneid * 2.0f);
    A[laneid + 32 + 32]=__float2half(laneid * 3.0f);
    A[laneid + 32 + 32 + 32]=__float2half(laneid * 4.0f);
    __syncthreads();

    if(laneid == 0){
        dumpEx_row_major<half>(A, row,col, "16bit A:",1,2);
    }

    uint32_t RA[2];

    LDMATRIX_TRANS_X2(RA[0], RA[1], __cvta_generic_to_shared(&A[(laneid * 8) % (row * col)]));


    int x = static_cast<int>(RA[0]);
    int y = static_cast<int>(RA[1]);

    // 1. 将 uint32_t 拆分为两个 uint16_t
    uint16_t high16_x = static_cast<uint16_t>(x >> 16);  // 高16位
    uint16_t low16_x  = static_cast<uint16_t>(x & 0xFFFF); // 低16位
    uint16_t high16_y = static_cast<uint16_t>(y >> 16);  // 高16位
    uint16_t low16_y  = static_cast<uint16_t>(y & 0xFFFF); // 低16位
    
    // 2. 将 uint16_t 的位模式重新解释为 half 类型
    __half h_high_x, h_low_x;
    __half h_high_y, h_low_y;

    // 使用 memcpy 或类型双关来保持位模式不变
    memcpy(&h_high_x, &high16_x, sizeof(uint16_t));
    memcpy(&h_low_x, &low16_x, sizeof(uint16_t));
    memcpy(&h_high_y, &high16_y, sizeof(uint16_t));
    memcpy(&h_low_y, &low16_y, sizeof(uint16_t));

    if (laneid == 0) {
        printf("==============================================PTX LDMATRX=====================================\n");
        printf("| LaneId | Variable     | Raw Hex  | Float Value  | | Variable     | Raw Hex  | Float Value  |\n");
        printf("|--------|--------------|----------|--------------| |--------------|----------|--------------|\n");
    }

    // 所有线程同步，确保表头先打印（如果多个线程都打印表头会混乱，所以只让一个线程打印表头）
    __syncthreads(); // 如果这是在block级别，需要同步

    // 每个线程打印自己的信息，但将X的高16位和低16位打印在同一行
    printf("| %6d | X (Low 16)  |   0x%04X | %12.6f | "
        "| X (High 16)   |   0x%04X | %12.6f |\n", 
        static_cast<int>(laneid), 
        low16_x, __half2float(h_low_x),
        high16_x, __half2float(h_high_x));
    printf("| %6d | Y (Low 16)  |   0x%04X | %12.6f | "
        "| Y (High 16)   |   0x%04X | %12.6f |\n", 
        static_cast<int>(laneid), 
        low16_y, __half2float(h_low_y),
        high16_y, __half2float(h_high_y));

}

// cuda替换ptx-ldmatirx.m8n8.trans.x2
__global__ void load_m8n8_trans_x2(){
    constexpr int row = 16;
    constexpr int col = 8;
    __shared__ half A[row * col];
    const int laneid = (threadIdx.z * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x) 
                        & (warpSize - 1);
    A[laneid]=__float2half(laneid * 1.0f);
    A[laneid + 32]=__float2half(laneid * 2.0f);
    A[laneid + 32 + 32]=__float2half(laneid * 3.0f);
    A[laneid + 32 + 32 + 32]=__float2half(laneid * 4.0f);
    __syncthreads();

    if(laneid == 0){
        dumpEx_row_major<half>(A, row,col, "16bit A:",1,2);
    }

    int index_temp =   (laneid * 8) % (row * col);

    printf("Laneid: %d, index_temp: %d\n", laneid, index_temp);

    uint32_t *ptr = reinterpret_cast<uint32_t*>(&A[index_temp]);

    __half2 h2_x = {0,0};
    __half2 h2_y = {0,0};


    uintptr_t temp_x = 0;
    uintptr_t temp_y = 0;

    #pragma unroll 4
    for(int i = 0; i < 4; i++) {
        temp_x = __shfl_sync(0xFFFFFFFF, reinterpret_cast<uintptr_t>(ptr), 2 * i);
        temp_y = __shfl_sync(0xFFFFFFFF, reinterpret_cast<uintptr_t>(ptr), 2 * i + 8);
        if(laneid % 4 == i){
            h2_x.x = *(reinterpret_cast<half*>(temp_x) + (laneid / 4));
            h2_y.x = *(reinterpret_cast<half*>(temp_y) + (laneid / 4));
        }
        temp_x = __shfl_sync(0xFFFFFFFF, reinterpret_cast<uintptr_t>(ptr), 2 * i + 1);
        temp_y = __shfl_sync(0xFFFFFFFF, reinterpret_cast<uintptr_t>(ptr), 2 * i + 1 + 8);
        if(laneid % 4 == i){
            h2_x.y = *(reinterpret_cast<half*>(temp_x) + (laneid / 4));            
            h2_y.y = *(reinterpret_cast<half*>(temp_y) + (laneid / 4));            
        }
    }
    int x = *reinterpret_cast<int*>(&h2_x);
    int y = *reinterpret_cast<int*>(&h2_y);

    // 1. 将 uint32_t 拆分为两个 uint16_t
    uint16_t high16_x = static_cast<uint16_t>(x >> 16);  // 高16位
    uint16_t low16_x  = static_cast<uint16_t>(x & 0xFFFF); // 低16位
    uint16_t high16_y = static_cast<uint16_t>(y >> 16);  // 高16位
    uint16_t low16_y  = static_cast<uint16_t>(y & 0xFFFF); // 低16位
    // 2. 将 uint16_t 的位模式重新解释为 half 类型
    __half h_high_x, h_low_x;
    __half h_high_y, h_low_y;
    // 使用 memcpy 或类型双关来保持位模式不变
    memcpy(&h_high_x, &high16_x, sizeof(uint16_t));
    memcpy(&h_low_x, &low16_x, sizeof(uint16_t));
    memcpy(&h_high_y, &high16_y, sizeof(uint16_t));
    memcpy(&h_low_y, &low16_y, sizeof(uint16_t));


    if (laneid == 0) {
        printf("===============================================SOFT LOAD======================================\n");
        printf("| LaneId | Variable     | Raw Hex  | Float Value  | | Variable     | Raw Hex  | Float Value  |\n");
        printf("|--------|--------------|----------|--------------| |--------------|----------|--------------|\n");
    }

    // 所有线程同步，确保表头先打印（如果多个线程都打印表头会混乱，所以只让一个线程打印表头）
    __syncthreads(); // 如果这是在block级别，需要同步

    // 每个线程打印自己的信息，但将X的高16位和低16位打印在同一行
    printf("| %6d | X (Low 16)  |   0x%04X | %12.6f | "
        "| X (High 16)   |   0x%04X | %12.6f |\n", 
        static_cast<int>(laneid), 
        low16_x, __half2float(h_low_x),
        high16_x, __half2float(h_high_x));
    printf("| %6d | Y (Low 16)  |   0x%04X | %12.6f | "
        "| Y (High 16)   |   0x%04X | %12.6f |\n", 
        static_cast<int>(laneid), 
        low16_y, __half2float(h_low_y),
        high16_y, __half2float(h_high_y));
}
// ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
//                                 ldmatrix.X4                       
// ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
__global__ void ldmatrix_m8n8_x4(){
    __shared__ half A[32*8];
    const size_t laneid = threadIdx.x % 32;
    A[laneid]=__float2half(laneid * 1.0f);
    A[laneid + 32]=__float2half(laneid * 2.0f);
    A[laneid + 32 + 32]=__float2half(laneid * 3.0f);
    A[laneid + 32 + 32 + 32]=__float2half(laneid * 4.0f);
    A[laneid + 32 + 32 + 32 + 32]=__float2half(laneid * 5.0f);
    A[laneid + 32 + 32 + 32 + 32 + 32]=__float2half(laneid * 6.0f);
    A[laneid + 32 + 32 + 32 + 32 + 32 + 32]=__float2half(laneid * 7.0f);
    A[laneid + 32 + 32 + 32 + 32 + 32 + 32 + 32]=__float2half(laneid * 8.0f);
    __syncthreads();

    uint32_t RA[4];

    LDMATRIX_X4(RA[0], RA[1], RA[2], RA[3], __cvta_generic_to_shared(&A[(laneid * 8) % (32 * 8)]));


}

// cuda替换ptx-ldmatirx.m8n8.x4
__global__ void load_m8n8_x4(){
    __shared__ half A[32*8];
    const int laneid = (threadIdx.z * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x) 
                        & (warpSize - 1);
    A[laneid]=__float2half(laneid * 1.0f);
    A[laneid + 32]=__float2half(laneid * 2.0f);
    A[laneid + 32 + 32]=__float2half(laneid * 3.0f);
    A[laneid + 32 + 32 + 32]=__float2half(laneid * 4.0f);
    A[laneid + 32 + 32 + 32 + 32]=__float2half(laneid * 5.0f);
    A[laneid + 32 + 32 + 32 + 32 + 32]=__float2half(laneid * 6.0f);
    A[laneid + 32 + 32 + 32 + 32 + 32 + 32]=__float2half(laneid * 7.0f);
    A[laneid + 32 + 32 + 32 + 32 + 32 + 32 + 32]=__float2half(laneid * 8.0f);
    __syncthreads();

    if(laneid == 0){
        dumpEx_row_major<half>(A, 32,8, "16bit A:",1,2);
    }

    uint32_t *ptr = reinterpret_cast<uint32_t*>(&A[(laneid * 8) % (32 * 8)]);

    uint32_t *cur_ptr_x = nullptr;
    uint32_t *cur_ptr_y = nullptr;
    uint32_t *cur_ptr_z = nullptr;
    uint32_t *cur_ptr_w = nullptr;
    #pragma unroll 32// 指示编译器将后续循环展开16次
    for(int i=0;i<32;i++){
        uintptr_t temp;
        temp = __shfl_sync(0xFFFFFFFF, reinterpret_cast<uintptr_t>(ptr), i);
        if(laneid >= i * 4&& laneid < (i + 1) * 4){
            cur_ptr_x = reinterpret_cast<uint32_t*>(temp) + (laneid % 4);

        }else if(laneid >= (i - 8) * 4 && laneid < (i + 1 - 8) * 4){
            cur_ptr_y = reinterpret_cast<uint32_t*>(temp) + (laneid % 4);
        }
        else if(laneid >= (i - 16) * 4 && laneid < (i + 1 - 16) * 4){
            cur_ptr_z = reinterpret_cast<uint32_t*>(temp) + (laneid % 4);
        }
        else if(laneid >= (i - 24) * 4 && laneid < (i + 1 - 24) * 4){
            cur_ptr_w = reinterpret_cast<uint32_t*>(temp) + (laneid % 4);
        }
    }

    // printf("Device: cur_ptr_x = %p  value = %f  LaneId=%d\n", cur_ptr_x, __half2float(*(half*)cur_ptr_x), laneid);
    // printf("Device: cur_ptr_y = %p  value = %f  LaneId=%d\n", cur_ptr_y, __half2float(*(half*)cur_ptr_y), laneid);

    int x = *cur_ptr_x;
    int y = *cur_ptr_y;
    int z = *cur_ptr_z;
    int w = *cur_ptr_w;
    

    // 1. 将 uint32_t 拆分为两个 uint16_t
    uint16_t high16_x = static_cast<uint16_t>(x >> 16);  // 高16位
    uint16_t low16_x  = static_cast<uint16_t>(x & 0xFFFF); // 低16位
    uint16_t high16_y = static_cast<uint16_t>(y >> 16);  // 高16位
    uint16_t low16_y  = static_cast<uint16_t>(y & 0xFFFF); // 低16位
    uint16_t high16_z = static_cast<uint16_t>(z >> 16);  // 高16位
    uint16_t low16_z  = static_cast<uint16_t>(z & 0xFFFF); // 低16位
    uint16_t high16_w = static_cast<uint16_t>(w >> 16);  // 高16位
    uint16_t low16_w  = static_cast<uint16_t>(w & 0xFFFF); // 低16位
    // 2. 将 uint16_t 的位模式重新解释为 half 类型
    __half h_high_x, h_low_x;
    __half h_high_y, h_low_y;
    __half h_high_z, h_low_z;
    __half h_high_w, h_low_w;
    // 使用 memcpy 或类型双关来保持位模式不变
    memcpy(&h_high_x, &high16_x, sizeof(uint16_t));
    memcpy(&h_low_x, &low16_x, sizeof(uint16_t));
    memcpy(&h_high_y, &high16_y, sizeof(uint16_t));
    memcpy(&h_low_y, &low16_y, sizeof(uint16_t));
    memcpy(&h_high_z, &high16_z, sizeof(uint16_t));
    memcpy(&h_low_z, &low16_z, sizeof(uint16_t));
    memcpy(&h_high_w, &high16_w, sizeof(uint16_t));
    memcpy(&h_low_w, &low16_w, sizeof(uint16_t));

    if (laneid == 0) {
        printf("===============================================SOFT LOAD======================================\n");
        printf("| LaneId | Variable     | Raw Hex  | Float Value  | | Variable     | Raw Hex  | Float Value  |\n");
        printf("|--------|--------------|----------|--------------| |--------------|----------|--------------|\n");
    }

    // 所有线程同步，确保表头先打印（如果多个线程都打印表头会混乱，所以只让一个线程打印表头）
    __syncthreads(); // 如果这是在block级别，需要同步

    // 每个线程打印自己的信息，但将X的高16位和低16位打印在同一行
    printf("| %6d | X (Low 16)  |   0x%04X | %12.6f | "
        "| X (High 16)   |   0x%04X | %12.6f |\n", 
        static_cast<int>(laneid), 
        low16_x, __half2float(h_low_x),
        high16_x, __half2float(h_high_x));
    printf("| %6d | Y (Low 16)  |   0x%04X | %12.6f | "
        "| Y (High 16)   |   0x%04X | %12.6f |\n", 
        static_cast<int>(laneid), 
        low16_y, __half2float(h_low_y),
        high16_y, __half2float(h_high_y));
    printf("| %6d | Z (Low 16)  |   0x%04X | %12.6f | "
        "| Z (High 16)   |   0x%04X | %12.6f |\n", 
        static_cast<int>(laneid), 
        low16_z, __half2float(h_low_z),
        high16_z, __half2float(h_high_z));
    printf("| %6d | W (Low 16)  |   0x%04X | %12.6f | "
        "| W (High 16)   |   0x%04X | %12.6f |\n", 
        static_cast<int>(laneid), 
        low16_w, __half2float(h_low_w),
        high16_w, __half2float(h_high_w));
}
// ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
//                            ldmatrix.trans.X4                       
// ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
__global__ void ldmatrix_m8n8_trans_x4(){
    constexpr int row = 32;
    constexpr int col = 8;
    __shared__ half A[row * col];
    const int laneid = (threadIdx.z * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x) 
                        & (warpSize - 1);
    A[laneid]=__float2half(laneid * 1.0f);
    A[laneid + 32]=__float2half(laneid * 2.0f);
    A[laneid + 32 + 32]=__float2half(laneid * 3.0f);
    A[laneid + 32 + 32 + 32]=__float2half(laneid * 4.0f);
    A[laneid + 32 + 32 + 32 + 32]=__float2half(laneid * 5.0f);
    A[laneid + 32 + 32 + 32 + 32 + 32]=__float2half(laneid * 6.0f);
    A[laneid + 32 + 32 + 32 + 32 + 32 + 32]=__float2half(laneid * 7.0f);
    A[laneid + 32 + 32 + 32 + 32 + 32 + 32 + 32]=__float2half(laneid * 8.0f);
    __syncthreads();

    if(laneid == 0){
        dumpEx_row_major<half>(A, row,col, "16bit A:",1,2);
    }

    uint32_t RA[4];

    LDMATRIX_TRANS_X4(RA[0], RA[1], RA[2], RA[3], __cvta_generic_to_shared(&A[(laneid * 8) % (row * col)]));

    int x = static_cast<int>(RA[0]);
    int y = static_cast<int>(RA[1]);
    int z = static_cast<int>(RA[2]);
    int w = static_cast<int>(RA[3]);

    // 1. 将 uint32_t 拆分为两个 uint16_t
    uint16_t high16_x = static_cast<uint16_t>(x >> 16);  // 高16位
    uint16_t low16_x  = static_cast<uint16_t>(x & 0xFFFF); // 低16位
    uint16_t high16_y = static_cast<uint16_t>(y >> 16);  // 高16位
    uint16_t low16_y  = static_cast<uint16_t>(y & 0xFFFF); // 低16位
    uint16_t high16_z = static_cast<uint16_t>(z >> 16);  // 高16位
    uint16_t low16_z  = static_cast<uint16_t>(z & 0xFFFF); // 低16位
    uint16_t high16_w = static_cast<uint16_t>(w >> 16);  // 高16位
    uint16_t low16_w  = static_cast<uint16_t>(w & 0xFFFF); // 低16位
    
    // 2. 将 uint16_t 的位模式重新解释为 half 类型
    __half h_high_x, h_low_x;
    __half h_high_y, h_low_y;
    __half h_high_z, h_low_z;
    __half h_high_w, h_low_w;

    // 使用 memcpy 或类型双关来保持位模式不变
    memcpy(&h_high_x, &high16_x, sizeof(uint16_t));
    memcpy(&h_low_x, &low16_x, sizeof(uint16_t));
    memcpy(&h_high_y, &high16_y, sizeof(uint16_t));
    memcpy(&h_low_y, &low16_y, sizeof(uint16_t));
    memcpy(&h_high_z, &high16_z, sizeof(uint16_t));
    memcpy(&h_low_z, &low16_z, sizeof(uint16_t));
    memcpy(&h_high_w, &high16_w, sizeof(uint16_t));
    memcpy(&h_low_w, &low16_w, sizeof(uint16_t));

    if (laneid == 0) {
        printf("==============================================PTX LDMATRX=====================================\n");
        printf("| LaneId | Variable     | Raw Hex  | Float Value  | | Variable     | Raw Hex  | Float Value  |\n");
        printf("|--------|--------------|----------|--------------| |--------------|----------|--------------|\n");
    }

    // 所有线程同步，确保表头先打印（如果多个线程都打印表头会混乱，所以只让一个线程打印表头）
    __syncthreads(); // 如果这是在block级别，需要同步

    // 每个线程打印自己的信息，但将X的高16位和低16位打印在同一行
    printf("| %6d | X (Low 16)  |   0x%04X | %12.6f | "
        "| X (High 16)   |   0x%04X | %12.6f |\n", 
        static_cast<int>(laneid), 
        low16_x, __half2float(h_low_x),
        high16_x, __half2float(h_high_x));
    printf("| %6d | Y (Low 16)  |   0x%04X | %12.6f | "
        "| Y (High 16)   |   0x%04X | %12.6f |\n", 
        static_cast<int>(laneid), 
        low16_y, __half2float(h_low_y),
        high16_y, __half2float(h_high_y));
    printf("| %6d | Z (Low 16)  |   0x%04X | %12.6f | "
        "| Z (High 16)   |   0x%04X | %12.6f |\n", 
        static_cast<int>(laneid), 
        low16_z, __half2float(h_low_z),
        high16_z, __half2float(h_high_z));
    printf("| %6d | W (Low 16)  |   0x%04X | %12.6f | "
        "| W (High 16)   |   0x%04X | %12.6f |\n", 
        static_cast<int>(laneid), 
        low16_w, __half2float(h_low_w),
        high16_w, __half2float(h_high_w));
}

// cuda替换ptx-ldmatirx.m8n8.x4
__global__ void load_m8n8_trans_x4(){
    constexpr int row = 32;
    constexpr int col = 8;
    __shared__ half A[row * col];
    const int laneid = (threadIdx.z * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x) 
                        & (warpSize - 1);
    A[laneid]=__float2half(laneid * 1.0f);
    A[laneid + 32]=__float2half(laneid * 2.0f);
    A[laneid + 32 + 32]=__float2half(laneid * 3.0f);
    A[laneid + 32 + 32 + 32]=__float2half(laneid * 4.0f);
    A[laneid + 32 + 32 + 32 + 32]=__float2half(laneid * 5.0f);
    A[laneid + 32 + 32 + 32 + 32 + 32]=__float2half(laneid * 6.0f);
    A[laneid + 32 + 32 + 32 + 32 + 32 + 32]=__float2half(laneid * 7.0f);
    A[laneid + 32 + 32 + 32 + 32 + 32 + 32 + 32]=__float2half(laneid * 8.0f);
    __syncthreads();

    if(laneid == 0){
        dumpEx_row_major<half>(A, row,col, "16bit A:",1,2);
    }

    int index_temp =   (laneid * 8) % (row * col);

    printf("Laneid: %d, index_temp: %d\n", laneid, index_temp);

    uint32_t *ptr = reinterpret_cast<uint32_t*>(&A[index_temp]);

    __half2 h2_x = {0,0};
    __half2 h2_y = {0,0};
    __half2 h2_z = {0,0};
    __half2 h2_w = {0,0};


    uintptr_t temp_x = 0;
    uintptr_t temp_y = 0;
    uintptr_t temp_z = 0;
    uintptr_t temp_w = 0;

    #pragma unroll 4
    for(int i = 0; i < 4; i++) {
        temp_x = __shfl_sync(0xFFFFFFFF, reinterpret_cast<uintptr_t>(ptr), 2 * i);
        temp_y = __shfl_sync(0xFFFFFFFF, reinterpret_cast<uintptr_t>(ptr), 2 * i + 8);
        temp_z = __shfl_sync(0xFFFFFFFF, reinterpret_cast<uintptr_t>(ptr), 2 * i + 8 + 8);
        temp_w = __shfl_sync(0xFFFFFFFF, reinterpret_cast<uintptr_t>(ptr), 2 * i + 8 + 8 + 8);
        if(laneid % 4 == i){
            h2_x.x = *(reinterpret_cast<half*>(temp_x) + (laneid / 4));
            h2_y.x = *(reinterpret_cast<half*>(temp_y) + (laneid / 4));
            h2_z.x = *(reinterpret_cast<half*>(temp_z) + (laneid / 4));
            h2_w.x = *(reinterpret_cast<half*>(temp_w) + (laneid / 4));
        }
        temp_x = __shfl_sync(0xFFFFFFFF, reinterpret_cast<uintptr_t>(ptr), 2 * i + 1);
        temp_y = __shfl_sync(0xFFFFFFFF, reinterpret_cast<uintptr_t>(ptr), 2 * i + 1 + 8);
        temp_z = __shfl_sync(0xFFFFFFFF, reinterpret_cast<uintptr_t>(ptr), 2 * i + 1 + 8 + 8);
        temp_w = __shfl_sync(0xFFFFFFFF, reinterpret_cast<uintptr_t>(ptr), 2 * i + 1 + 8 + 8 + 8);
        if(laneid % 4 == i){
            h2_x.y = *(reinterpret_cast<half*>(temp_x) + (laneid / 4));            
            h2_y.y = *(reinterpret_cast<half*>(temp_y) + (laneid / 4));            
            h2_z.y = *(reinterpret_cast<half*>(temp_z) + (laneid / 4));            
            h2_w.y = *(reinterpret_cast<half*>(temp_w) + (laneid / 4));            
        }
    }
    int x = *reinterpret_cast<int*>(&h2_x);
    int y = *reinterpret_cast<int*>(&h2_y);
    int z = *reinterpret_cast<int*>(&h2_z);
    int w = *reinterpret_cast<int*>(&h2_w);


    // 1. 将 uint32_t 拆分为两个 uint16_t
    uint16_t high16_x = static_cast<uint16_t>(x >> 16);  // 高16位
    uint16_t low16_x  = static_cast<uint16_t>(x & 0xFFFF); // 低16位
    uint16_t high16_y = static_cast<uint16_t>(y >> 16);  // 高16位
    uint16_t low16_y  = static_cast<uint16_t>(y & 0xFFFF); // 低16位
    uint16_t high16_z = static_cast<uint16_t>(z >> 16);  // 高16位
    uint16_t low16_z  = static_cast<uint16_t>(z & 0xFFFF); // 低16位
    uint16_t high16_w = static_cast<uint16_t>(w >> 16);  // 高16位
    uint16_t low16_w  = static_cast<uint16_t>(w & 0xFFFF); // 低16位
    // 2. 将 uint16_t 的位模式重新解释为 half 类型
    __half h_high_x, h_low_x;
    __half h_high_y, h_low_y;
    __half h_high_z, h_low_z;
    __half h_high_w, h_low_w;
    // 使用 memcpy 或类型双关来保持位模式不变
    memcpy(&h_high_x, &high16_x, sizeof(uint16_t));
    memcpy(&h_low_x, &low16_x, sizeof(uint16_t));
    memcpy(&h_high_y, &high16_y, sizeof(uint16_t));
    memcpy(&h_low_y, &low16_y, sizeof(uint16_t));
    memcpy(&h_high_z, &high16_z, sizeof(uint16_t));
    memcpy(&h_low_z, &low16_z, sizeof(uint16_t));
    memcpy(&h_high_w, &high16_w, sizeof(uint16_t));
    memcpy(&h_low_w, &low16_w, sizeof(uint16_t));

    if (laneid == 0) {
        printf("===============================================SOFT LOAD======================================\n");
        printf("| LaneId | Variable     | Raw Hex  | Float Value  | | Variable     | Raw Hex  | Float Value  |\n");
        printf("|--------|--------------|----------|--------------| |--------------|----------|--------------|\n");
    }

    // 所有线程同步，确保表头先打印（如果多个线程都打印表头会混乱，所以只让一个线程打印表头）
    __syncthreads(); // 如果这是在block级别，需要同步

    // 每个线程打印自己的信息，但将X的高16位和低16位打印在同一行
    printf("| %6d | X (Low 16)  |   0x%04X | %12.6f | "
        "| X (High 16)   |   0x%04X | %12.6f |\n", 
        static_cast<int>(laneid), 
        low16_x, __half2float(h_low_x),
        high16_x, __half2float(h_high_x));
    printf("| %6d | Y (Low 16)  |   0x%04X | %12.6f | "
        "| Y (High 16)   |   0x%04X | %12.6f |\n", 
        static_cast<int>(laneid), 
        low16_y, __half2float(h_low_y),
        high16_y, __half2float(h_high_y));
    printf("| %6d | Z (Low 16)  |   0x%04X | %12.6f | "
        "| Z (High 16)   |   0x%04X | %12.6f |\n", 
        static_cast<int>(laneid), 
        low16_z, __half2float(h_low_z),
        high16_z, __half2float(h_high_z));
    printf("| %6d | W (Low 16)  |   0x%04X | %12.6f | "
        "| W (High 16)   |   0x%04X | %12.6f |\n", 
        static_cast<int>(laneid), 
        low16_w, __half2float(h_low_w),
        high16_w, __half2float(h_high_w));
}

int main(){

    ldmatrix_m8n16_4bit_x1<<<1,32>>>();
    cudaError_t syncError = cudaDeviceSynchronize();
    if (syncError != cudaSuccess) {
    
        printf("ldmatrix_m8n16_4bit_x1 failed with error: %s\n", cudaGetErrorString(syncError));
    }
    #if defined (M8N8_X1)
    ldmatrix_m8n8_x1<<<1,32>>>();
    syncError = cudaDeviceSynchronize();
    if (syncError != cudaSuccess) {
        printf("ldmatrix_m8n8_x1 failed with error: %s\n", cudaGetErrorString(syncError));
    }    

    load_m8n8_x1<<<1,32>>>();
    syncError = cudaDeviceSynchronize();
    if (syncError != cudaSuccess) {
        printf("load_m8n8_x1 failed with error: %s\n", cudaGetErrorString(syncError));
    }    
    #endif

    #if defined (M8N8_X2)
    ldmatrix_m8n8_x2<<<1,32>>>();
    syncError = cudaDeviceSynchronize();
    if (syncError != cudaSuccess) {
        printf("ldmatrix_m8n8_x2 failed with error: %s\n", cudaGetErrorString(syncError));
    }  
    
    load_m8n8_x2<<<1,32>>>();
    syncError = cudaDeviceSynchronize();
    if (syncError != cudaSuccess) {
        printf("load_m8n8_x2 failed with error: %s\n", cudaGetErrorString(syncError));
    }  
    #endif


    #if defined (M8N8_X4)
    ldmatrix_m8n8_x4<<<1,32>>>();
    syncError = cudaDeviceSynchronize();
    if (syncError != cudaSuccess) {
        printf("ldmatrix_m8n8_x4 failed with error: %s\n", cudaGetErrorString(syncError));
    }    


    load_m8n8_x4<<<1,32>>>();
    syncError = cudaDeviceSynchronize();
    if (syncError != cudaSuccess) {
        printf("load_m8n8_x4 failed with error: %s\n", cudaGetErrorString(syncError));
    }    

    #endif

    // 转置
    #if defined (M8N8_TRANS_X1)
    ldmatrix_m8n8_trans_x1<<<1,32>>>();
    syncError = cudaDeviceSynchronize();
    if (syncError != cudaSuccess) {
        printf("ldmatrix_m8n8_trans_x1 failed with error: %s\n", cudaGetErrorString(syncError));
    }    

    load_m8n8_trans_x1<<<1,32>>>();
    syncError = cudaDeviceSynchronize();
    if (syncError != cudaSuccess) {
        printf("load_m8n8_trans_x1 failed with error: %s\n", cudaGetErrorString(syncError));
    }    
    #endif
    
    
    #if defined (M8N8_TRANS_X2)
    ldmatrix_m8n8_trans_x2<<<1,32>>>();
    syncError = cudaDeviceSynchronize();
    if (syncError != cudaSuccess) {
        printf("ldmatrix_m8n8_trans_x2 failed with error: %s\n", cudaGetErrorString(syncError));
    }    

    load_m8n8_trans_x2<<<1,32>>>();
    syncError = cudaDeviceSynchronize();
    if (syncError != cudaSuccess) {
        printf("load_m8n8_trans_x2 failed with error: %s\n", cudaGetErrorString(syncError));
    }    
    #endif


    #if defined (M8N8_TRANS_X4)
    ldmatrix_m8n8_trans_x4<<<1,32>>>();
    syncError = cudaDeviceSynchronize();
    if (syncError != cudaSuccess) {
        printf("ldmatrix_m8n8_trans_x4 failed with error: %s\n", cudaGetErrorString(syncError));
    }    

    load_m8n8_trans_x4<<<1,32>>>();
    syncError = cudaDeviceSynchronize();
    if (syncError != cudaSuccess) {
        printf("load_m8n8_trans_x4 failed with error: %s\n", cudaGetErrorString(syncError));
    }    
    #endif

    printf("Done\n");
   
    return 0;

}