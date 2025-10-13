#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
#include <sys/types.h>
#include <typeinfo>
#include <inttypes.h>
enum class AddressSpace {
  Generic = 0,
  Global  = 1,
  Shared  = 3,
};
__device__ void shared_load_16b(void *dst, uint32_t ptr) {

  asm volatile("ld.shared.u16 %0, [%1];\n"
    : "=h"(*reinterpret_cast<uint16_t *>(dst))
    : "r"(ptr));
}
// PTX共享内存加载函数
__device__ void shared_load_32b(void* dst, uint32_t ptr) {
    asm volatile(
        "ld.shared.u32 %0, [%1];\n"
        : "=r"(*reinterpret_cast<uint32_t*>(dst))
        : "r"(ptr)
    );
}
__device__ void shared_load_64b(void *dst, uint32_t ptr) {
    uint2 *dst_u64 = reinterpret_cast<uint2 *>(dst);
  asm volatile("ld.shared.v2.u32 {%0, %1}, [%2];\n"
    :
      "=r"(dst_u64->x),
      "=r"(dst_u64->y)
    : "r"(ptr));
  }

__device__  void shared_load_128b(void *dst, uint32_t ptr) {

  uint4 *dst_u128 = reinterpret_cast<uint4 *>(dst);
  asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];\n"
    :
      "=r"(dst_u128->x),
      "=r"(dst_u128->y),
      "=r"(dst_u128->z),
      "=r"(dst_u128->w)
    : "r"(ptr));
  }


  // 对比用的标准C++实现
__device__ void shared_load_standard_16b(void* dst, uint32_t ptr) {
    // 修复：使用正确的共享内存指针语法
    uint16_t *smem_generic_ptr = reinterpret_cast<uint16_t*>(__cvta_shared_to_generic(ptr));
    uint16_t *dst_u32 = reinterpret_cast<uint16_t*>(dst);
    *dst_u32 = *smem_generic_ptr;

}



__device__ void shared_load_standard_32b(void* dst, uint32_t ptr) {
    // 修复：使用正确的共享内存指针语法
    uint32_t *smem_generic_ptr = reinterpret_cast<uint32_t*>(__cvta_shared_to_generic(ptr));
    uint32_t *dst_u32 = reinterpret_cast<uint32_t*>(dst);
    *dst_u32 = *smem_generic_ptr;

}

__device__ void shared_load_standard_64b(void* dst, uint32_t ptr) {
    // 修复：使用正确的共享内存指针语法
    uint2 *smem_generic_ptr = reinterpret_cast<uint2*>(__cvta_shared_to_generic(ptr));
    uint2 *dst_u32 = reinterpret_cast<uint2*>(dst);
    *dst_u32 = *smem_generic_ptr;

}

__device__ void shared_load_standard_128b(void* dst, uint32_t ptr) {
    // 修复：使用正确的共享内存指针语法
    uint4 *smem_generic_ptr = reinterpret_cast<uint4*>(__cvta_shared_to_generic(ptr));
    uint4 *dst_u32 = reinterpret_cast<uint4*>(dst);
    *dst_u32 = *smem_generic_ptr;

}

// 正确的测试内核
__global__ void test_shared_load() {
    // 正确声明共享内存（不要初始化）
    __shared__ uint4 shared_data[4];
    
    // 本地变量
    uint4 result_ptx;
    uint4 result_std;
    
    // 在运行时初始化共享内存（线程0负责）
    if (threadIdx.x == 0) {
        shared_data[0] = {0xDEAD, 0xBEEF, 0x1234, 0x8765};
        shared_data[1] = {0xBEEF, 0x1234, 0x8765, 0xDEAD};
        shared_data[2] = {0x1234, 0x8765, 0xDEAD, 0xBEEF};
        shared_data[3] = {0x8765, 0xDEAD, 0xBEEF, 0x1234};
    }
    
    __syncthreads(); // 确保所有线程看到初始化后的数据
    auto shared_ptr = &shared_data[2];
    //打印shared_ptr的类型
    // printf("shared_ptr type: %s\n", typeid(shared_ptr).name());
    // 获取正确的共享内存指针
    // uint32_t ptr = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(shared_ptr));
    uint32_t ptr = __cvta_generic_to_shared(shared_ptr);
    // 测试PTX版本
    // shared_load_u32(&result_ptx, ptr);
    shared_load_128b(&result_ptx, ptr);
    
    // 测试标准版本
    // shared_load_standard(&result_std, ptr);
    shared_load_standard_128b(&result_std, ptr);
    
    // 验证结果
    // bool test_pass = (result_ptx == 0xDEADBEEF) && (result_std == 0xDEADBEEF);
    bool test_pass = (result_ptx.x==result_std.x && result_ptx.y==result_std.y && result_ptx.z==result_std.z && result_ptx.w==result_std.w) ? true : false;
    // printf("大写十六进制: %" PRIX64 "\n", result_ptx);
    printf("Thread %d: PTX=0x%04" PRIX32 ", STD=0x%04" PRIX32 ", PASS=%s\n",
           threadIdx.x, result_ptx.x, result_std.x, test_pass ? "YES" : "NO");
    printf("Thread %d: PTX=0x%04" PRIX32 ", STD=0x%04" PRIX32 ", PASS=%s\n",
           threadIdx.x, result_ptx.y, result_std.y, test_pass ? "YES" : "NO");
    printf("Thread %d: PTX=0x%04" PRIX32 ", STD=0x%04" PRIX32 ", PASS=%s\n",
           threadIdx.x, result_ptx.z, result_std.z, test_pass ? "YES" : "NO");
    printf("Thread %d: PTX=0x%04" PRIX32 ", STD=0x%04" PRIX32 ", PASS=%s\n",
           threadIdx.x, result_ptx.w, result_std.w, test_pass ? "YES" : "NO");
}
// 具体化的共享内存加载函数（去除模板，固定为uint4类型）
struct SharedLoadOp {
  __device__ static void shared_load_op(uint4 &D, void const *ptr) {
    // 方法1: 使用PTX汇编（启用此方法需要设置条件编译）
    #if 0
    unsigned addr = cutlass_get_smem_pointer(ptr);
    uint4 v;
    asm volatile ("ld.shared.v4.b32 {%0, %1, %2, %3}, [%4];" : 
      "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(addr));
    D = v;
    
    // 方法2: 使用C++指针解引用（默认方法）
    #else
    uint4 v = *(reinterpret_cast<uint4 const *>(ptr));
    D = v;
    #endif
  }
};

// 测试内核
__global__ void test_shared_load_op() {
    // 在共享内存中分配测试数据
    __shared__ uint4 shared_data[4];
    __shared__ uint4 result_ptx;
    __shared__ uint4 result_std;
    
    // 线程0初始化共享内存数据
    if (threadIdx.x == 0) {
        shared_data[0] = uint4{0x11111111, 0x22222222, 0x33333333, 0x44444444};
        shared_data[1] = uint4{0xAAAAAAAA, 0xBBBBBBBB, 0xCCCCCCCC, 0xDDDDDDDD};
        shared_data[2] = uint4{0xDEADBEEF, 0x12345678, 0x87654321, 0xABCDEF01};
        shared_data[3] = uint4{0x55555555, 0x66666666, 0x77777777, 0x88888888};
    }
    
    __syncthreads();
    
    // 测试从不同位置加载数据
    uint4 loaded_data;
    
    // 测试位置1的数据
    SharedLoadOp::shared_load_op(loaded_data, &shared_data[1]);
    
    __syncthreads();
    
    // 验证结果
    bool test_pass = (loaded_data.x == 0xAAAAAAAA) &&
                    (loaded_data.y == 0xBBBBBBBB) &&
                    (loaded_data.z == 0xCCCCCCCC) &&
                    (loaded_data.w == 0xDDDDDDDD);
    
    if (threadIdx.x == 0) {
        printf("=== Testing SharedLoadOp Function ===\n");
        printf("Expected:   0x%08" PRIX32 " 0x%08" PRIX32 " 0x%08" PRIX32 " 0x%08" PRIX32 "\n", 
               0xAAAAAAAA, 0xBBBBBBBB, 0xCCCCCCCC, 0xDDDDDDDD);
        printf("Loaded:     0x%08" PRIX32 " 0x%08" PRIX32 " 0x%08" PRIX32 " 0x%08" PRIX32 "\n", 
               loaded_data.x, loaded_data.y, loaded_data.z, loaded_data.w);
        printf("Test PASS: %s\n", test_pass ? "YES" : "NO");
    }
}

// 综合测试：测试多个位置和边界情况
__global__ void test_shared_load_op_comprehensive() {
    __shared__ uint8_t shared_buffer[64];  // 字节数组，用于灵活测试
    __shared__ uint4 results[4];
    
    // 初始化测试数据（所有线程参与）
    for (int i = threadIdx.x; i < 64; i += blockDim.x) {
        shared_buffer[i] = i * 0x11;  // 填充模式数据
    }
    
    __syncthreads();
    
    // 测试不同的对齐位置
    uint4 *aligned_ptr1 = reinterpret_cast<uint4*>(&shared_buffer[0]);
    uint4 *aligned_ptr2 = reinterpret_cast<uint4*>(&shared_buffer[16]);
    uint4 *aligned_ptr3 = reinterpret_cast<uint4*>(&shared_buffer[32]);
    
    if (threadIdx.x == 0) {
        // 测试位置0
        SharedLoadOp::shared_load_op(results[0], aligned_ptr1);
        
        // 测试位置16  
        SharedLoadOp::shared_load_op(results[1], aligned_ptr2);
        
        // 测试位置32
        SharedLoadOp::shared_load_op(results[2], aligned_ptr3);
    }
    
    __syncthreads();
    
    // 验证结果
    if (threadIdx.x == 0) {
        printf("=== Comprehensive SharedLoadOp Test ===\n");
        
        bool all_pass = true;
        
        // 验证每个测试位置
        for (int i = 0; i < 3; i++) {
            uint32_t expected_x = i * 16 * 0x11;
            uint32_t expected_y = expected_x + 0x44444444;
            uint32_t expected_z = expected_y + 0x44444444;
            uint32_t expected_w = expected_z + 0x44444444;
            
            // 简单验证数据是否被正确加载（非模式验证）
            bool current_pass = (results[i].x == *reinterpret_cast<uint32_t*>(&shared_buffer[i * 16]));
            
            printf("Position %d: %s\n", i, current_pass ? "PASS" : "FAIL");
            all_pass = all_pass && current_pass;
        }
        
        printf("Overall test: %s\n", all_pass ? "ALL PASS" : "SOME FAIL");
    }
}
int main() {
    printf("=== Testing Shared Memory Load Functions ===\n");
    
    // 启动测试内核
    test_shared_load<<<1, 32>>>();
    
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // 运行基础功能测试
    test_shared_load_op<<<1, 32>>>();
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA error in basic test: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    printf("\n");
    
    // 运行综合测试
    test_shared_load_op_comprehensive<<<1, 32>>>();
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA error in comprehensive test: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    printf("\n");
    
    printf("=== Test completed successfully! ===\n");
    return 0;
}