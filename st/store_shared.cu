#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
#include <inttypes.h>

// PTX共享内存存储函数 - 16位版本
__device__ void shared_store_16b(uint32_t ptr, const void *src) {
  asm volatile("st.shared.u16 [%0], %1;\n"
    : :
    "r"(ptr),
    "h"(*reinterpret_cast<uint16_t const *>(src))
  );
}

// PTX共享内存存储函数 - 32位版本
__device__ void shared_store_32b(uint32_t ptr, const void *src) {
  asm volatile("st.shared.u32 [%0], %1;\n"
    : :
    "r"(ptr),
    "r"(*reinterpret_cast<uint32_t const *>(src))
  );
}

// PTX共享内存存储函数 - 64位版本
__device__ void shared_store_64b(uint32_t ptr, const void *src) {
  uint2 const *src_u64 = reinterpret_cast<uint2 const *>(src);
  asm volatile("st.shared.v2.u32 [%0], {%1, %2};\n"
    : :
    "r"(ptr),
    "r"(src_u64->x),
    "r"(src_u64->y)
  );
}

// PTX共享内存存储函数 - 128位版本
__device__ void shared_store_128b(uint32_t ptr, const void *src) {
  uint4 const *src_u128 = reinterpret_cast<uint4 const *>(src);
  asm volatile("st.shared.v4.u32 [%0], {%1, %2, %3, %4};\n"
    : :
    "r"(ptr),
    "r"(src_u128->x),
    "r"(src_u128->y),
    "r"(src_u128->z),
    "r"(src_u128->w)
  );
}

// 对比用的标准C++实现
__device__ void shared_store_standard_16b(uint32_t ptr, const void *src) {
    uint16_t *smem_generic_ptr = reinterpret_cast<uint16_t*>(__cvta_shared_to_generic(ptr));
    const uint16_t *src_u16 = reinterpret_cast<const uint16_t*>(src);
    *smem_generic_ptr = *src_u16;
}

__device__ void shared_store_standard_32b(uint32_t ptr, const void *src) {
    uint32_t *smem_generic_ptr = reinterpret_cast<uint32_t*>(__cvta_shared_to_generic(ptr));
    const uint32_t *src_u32 = reinterpret_cast<const uint32_t*>(src);
    *smem_generic_ptr = *src_u32;
}

__device__ void shared_store_standard_64b(uint32_t ptr, const void *src) {
    uint2 *smem_generic_ptr = reinterpret_cast<uint2*>(__cvta_shared_to_generic(ptr));
    const uint2 *src_u64 = reinterpret_cast<const uint2*>(src);
    *smem_generic_ptr = *src_u64;
}

__device__ void shared_store_standard_128b(uint32_t ptr, const void *src) {
    uint4 *smem_generic_ptr = reinterpret_cast<uint4*>(__cvta_shared_to_generic(ptr));
    const uint4 *src_u128 = reinterpret_cast<const uint4*>(src);
    *smem_generic_ptr = *src_u128;
}


// 测试内核 - 16位版本（重点测试你的目标函数）
__global__ void test_shared_store_16b() {
    __shared__ uint16_t shared_data_ptx[8];
    __shared__ uint16_t shared_data_std[8];
    
    // 测试数据
    uint16_t test_data = 0xABCD;
    
    // 获取共享内存指针（测试不同偏移）
    uint32_t ptr_ptx = __cvta_generic_to_shared(&shared_data_ptx[2]);
    uint32_t ptr_std = __cvta_generic_to_shared(&shared_data_std[2]);
    
    __syncthreads();
    
    // 使用PTX版本存储
    shared_store_16b(ptr_ptx, &test_data);
    
    // 使用标准版本存储
    shared_store_standard_16b(ptr_std, &test_data);
    
    __syncthreads();
    
    // 验证结果
    bool test_pass = (shared_data_ptx[2] == test_data) && 
                    (shared_data_std[2] == test_data) &&
                    (shared_data_ptx[2] == shared_data_std[2]);
    
    if (threadIdx.x == 0) {
        printf("=== Testing 16-bit Shared Memory Store ===\n");
        printf("Expected: 0x%04" PRIX16 "\n", test_data);
        printf("PTX Result: 0x%04" PRIX16 "\n", shared_data_ptx[2]);
        printf("STD Result: 0x%04" PRIX16 "\n", shared_data_std[2]);
        printf("Test PASS: %s\n", test_pass ? "YES" : "NO");
    }
}

// 测试内核 - 32位版本
__global__ void test_shared_store_32b() {
    __shared__ uint32_t shared_data_ptx[4];
    __shared__ uint32_t shared_data_std[4];
    
    // 测试数据
    uint32_t test_data = 0xDEADBEEF;
    
    // 获取共享内存指针
    uint32_t ptr_ptx = __cvta_generic_to_shared(&shared_data_ptx[1]);
    uint32_t ptr_std = __cvta_generic_to_shared(&shared_data_std[1]);
    
    __syncthreads();
    
    // 使用PTX版本存储
    shared_store_32b(ptr_ptx, &test_data);
    
    // 使用标准版本存储
    shared_store_standard_32b(ptr_std, &test_data);
    
    __syncthreads();
    
    // 验证结果
    bool test_pass = (shared_data_ptx[1] == shared_data_std[1]) &&
                    (shared_data_ptx[1] == test_data);
    
    if (threadIdx.x == 0) {
        printf("=== Testing 32-bit Shared Memory Store ===\n");
        printf("Expected: 0x%08" PRIX32 "\n", test_data);
        printf("PTX Result: 0x%08" PRIX32 "\n", shared_data_ptx[1]);
        printf("STD Result: 0x%08" PRIX32 "\n", shared_data_std[1]);
        printf("Test PASS: %s\n", test_pass ? "YES" : "NO");
    }
}

// 测试内核 - 64位版本
__global__ void test_shared_store_64b() {
    __shared__ uint2 shared_data_ptx[4];
    __shared__ uint2 shared_data_std[4];
    
    // 测试数据
    uint2 test_data;
    test_data.x = 0xDEADBEEF;
    test_data.y = 0x12345678;
    
    // 获取共享内存指针（测试索引1的位置）
    uint32_t ptr_ptx = __cvta_generic_to_shared(&shared_data_ptx[1]);
    uint32_t ptr_std = __cvta_generic_to_shared(&shared_data_std[1]);
    
    __syncthreads();
    
    // 使用PTX版本存储
    shared_store_64b(ptr_ptx, &test_data);
    
    // 使用标准版本存储
    shared_store_standard_64b(ptr_std, &test_data);
    
    __syncthreads();
    
    // 验证结果
    bool test_pass = (shared_data_ptx[1].x == shared_data_std[1].x) &&
                    (shared_data_ptx[1].y == shared_data_std[1].y) &&
                    (shared_data_ptx[1].x == test_data.x) &&
                    (shared_data_ptx[1].y == test_data.y);
    
    if (threadIdx.x == 0) {
        printf("=== Testing 64-bit Shared Memory Store ===\n");
        printf("Expected:      0x%08" PRIX32 " 0x%08" PRIX32 "\n", test_data.x, test_data.y);
        printf("PTX Result:    0x%08" PRIX32 " 0x%08" PRIX32 "\n", 
               shared_data_ptx[1].x, shared_data_ptx[1].y);
        printf("STD Result:    0x%08" PRIX32 " 0x%08" PRIX32 "\n", 
               shared_data_std[1].x, shared_data_std[1].y);
        printf("Test PASS: %s\n", test_pass ? "YES" : "NO");
        
        // 额外验证：检查相邻内存位置是否被意外修改
        bool no_corruption = (shared_data_ptx[0].x == 0 && shared_data_ptx[0].y == 0) &&
                            (shared_data_ptx[2].x == 0 && shared_data_ptx[2].y == 0);
        printf("Memory integrity: %s\n", no_corruption ? "OK" : "CORRUPTED");
    }
}

// 测试内核 - 128位版本
__global__ void test_shared_store_128b() {
    __shared__ uint4 shared_data_ptx[4];
    __shared__ uint4 shared_data_std[4];
    
    // 测试数据
    uint4 test_data = {0xDEADBEEF, 0x12345678, 0x87654321, 0xABCDEF01};
    
    // 获取共享内存指针
    uint32_t ptr_ptx = __cvta_generic_to_shared(&shared_data_ptx[1]);
    uint32_t ptr_std = __cvta_generic_to_shared(&shared_data_std[1]);
    
    __syncthreads();
    
    // 使用PTX版本存储
    shared_store_128b(ptr_ptx, &test_data);
    
    // 使用标准版本存储
    shared_store_standard_128b(ptr_std, &test_data);
    
    __syncthreads();
    
    // 验证结果
    bool test_pass = (shared_data_ptx[1].x == shared_data_std[1].x) &&
                    (shared_data_ptx[1].y == shared_data_std[1].y) &&
                    (shared_data_ptx[1].z == shared_data_std[1].z) &&
                    (shared_data_ptx[1].w == shared_data_std[1].w);
    
    if (threadIdx.x == 0) {
        printf("=== Testing 128-bit Shared Memory Store ===\n");
        printf("PTX Result:  0x%08" PRIX32 " 0x%08" PRIX32 " 0x%08" PRIX32 " 0x%08" PRIX32 "\n",
               shared_data_ptx[1].x, shared_data_ptx[1].y, shared_data_ptx[1].z, shared_data_ptx[1].w);
        printf("STD Result:  0x%08" PRIX32 " 0x%08" PRIX32 " 0x%08" PRIX32 " 0x%08" PRIX32 "\n",
               shared_data_std[1].x, shared_data_std[1].y, shared_data_std[1].z, shared_data_std[1].w);
        printf("Test PASS: %s\n", test_pass ? "YES" : "NO");
    }
}


int main() {
    printf("=== Shared Memory Store Function Test Suite ===\n\n");
    
    // 运行16位测试
    test_shared_store_16b<<<1, 32>>>();
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA error in 16b test: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    printf("\n");

    // 运行32位测试
    test_shared_store_32b<<<1,32>>>();
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA error in 32b test: %s\n", cudaGetErrorString(err));
        return 1;
    }
    printf("\n");


    // 运行64位测试
    test_shared_store_64b<<<1,32>>>();
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA error in 64b test: %s\n", cudaGetErrorString(err));
        return 1;
    }
    printf("\n");

    // 运行128位测试
    test_shared_store_128b<<<1, 32>>>();
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA error in 128b test: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    printf("\n");
    
    
    printf("\n=== All tests completed! ===\n");
    return 0;
}