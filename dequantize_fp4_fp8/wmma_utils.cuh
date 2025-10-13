#include <stdio.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>
#include <cuda.h>
#include <mma.h>

#define CHECK_CUDA(status)                                                  \
{                                                                           \
    cudaError_t error = status;                                             \
    if (error != cudaSuccess) {                                             \
        std::cerr << "Got bad cuda status: " << cudaGetErrorString(error)   \
                << " at line: " << __LINE__ << std::endl;                   \
        exit(EXIT_FAILURE);                                                 \
    }                                                                       \
}

template<typename T>
void dump(T* data,const int M,const int N) {
    for(int r = 0; r < M; r++) {
        for(int c = 0; c < N; c++) {
            printf("%8.3f ", static_cast<float>(data[r * N + c]));
        }
        printf("\n");
    }
}

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

template<typename T>
__device__ __host__
void dumpEx_row_major(T* data, const int M, const int N, const char* name, 
            int group_row = 0, int group_col = 0) {
    printf("Dumping Row Major %s (Shape: %d x %d):\n", name, M, N);
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



template<typename T>
__device__ __host__
void dumpEx_col_major(T* data, const int M, const int N, const char* name, 
                      int group_row = 0, int group_col = 0) {
    printf("Dumping Col Major %s (Shape: %d x %d, Column-Major):\n", name, M, N);
    printf("     "); // 对齐列号标题
    
    // 打印列号标题 + 按列分组分隔符（逻辑不变）
    for (int c = 0; c < N; c++) {
        if (group_col > 0 && c % group_col == 0 && c != 0) 
            printf(" | ");
        printf("%8d ", c);
    }
    printf("\n");
    
    // 打印数据行（按行分组）
    for (int r = 0; r < M; r++) {
        // 行分组分隔符（逻辑不变）
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
            // 列分组分隔符（逻辑不变）
            if (group_col > 0 && c % group_col == 0 && c != 0) 
                printf(" | ");
            
            // 关键修改：按列主序访问数据
            printf("%8.3f ", static_cast<float>(data[c * M + r]));
        }
        printf("\n");
    }
}

// 打印32位二进制
__device__ __host__
void print_binary32(uint32_t val) {
    printf("uint32: ");
    for (int bit = 31; bit >= 0; --bit) {  // 从高位到低位 31~0
        printf("%c", (val >> bit) & 1 ? '1' : '0');
        if (bit % 4 == 0 && bit > 0) printf(" ");  // 每4位加空格
    }
    printf("\n");
}

// 打印16位二进制
__device__ __host__
void print_binary16(uint16_t val) {
    printf("16bit: ");
    for (int bit = 15; bit >= 0; --bit) {  // 从高位到低位 16~0
        printf("%c", (val >> bit) & 1 ? '1' : '0');
        if (bit % 4 == 0 && bit > 0) printf(" ");  // 每4位加空格
    }
    printf("\n");
}

// 打印4位二进制
__device__  __host__
void print_binary4(uint8_t val) {
    for (int bit = 3; bit >= 0; --bit) {  // 从高位到低位 3~0
        printf("%c", (val >> bit) & 1 ? '1' : '0');
    }
}

// 打印fp8的二进制
__device__ __host__
void print_binary_fp8(__nv_fp8_storage_t fp8__x) {
    printf("fp8: ");
    for (int bit = 7; bit >= 0; --bit) {  // 从高位到低位 7~0
        printf("%c", (fp8__x >> bit) & 1 ? '1' : '0');
        if (bit % 4 == 0 && bit > 0) printf(" ");  // 每4位加空格
    }
    printf("\n");
}