#include "../dequantize_fp4_fp8/wmma_utils.cuh"
#include "../dequantize_fp4_fp8/wmma.cuh"
#include "../m16n8k64_data.h"
#include <cstdint>
#include <cstdlib>
#include <sys/types.h>
#include <iostream>
#include <unordered_set>
#include <set>

using packed_a_t = uint4;
using packed_b_t = uint2;

using fp4x2 = __nv_fp4x2_e2m1;
using fp4x4 = __nv_fp4x4_e2m1;

using fp8 = __nv_fp8_e4m3;
struct alignas(32) packed_f32psum_t{
    float data[4];
};


constexpr int AM = 16;
constexpr int AN = 8;  
constexpr int AK = 64;


// 打包连续的一行8个fp4
__device__ __forceinline__
uint32_t pack_fp4(fp4 *fp4_ptr, int printLaneid = 0) {
    uint32_t packed = 0;
    const int laneid = threadIdx.x % WARP_SIZE;
    for (int i = 0; i < 8; ++i) {
        if(laneid == printLaneid){
            printf("Packing fp4[%d] = %8.3f ", i, static_cast<float>(fp4_ptr[i]));
            print_binary4(fp4_ptr[i].__x);
            printf("\n");
        }
        packed |= ((static_cast<uint32_t>(fp4_ptr[i].__x)) << (28 - 4 * i));
    }
    if(laneid == printLaneid) {
        printf("Packed value: ");
        print_binary32(packed);
        printf("\n");
    }
    return packed;
}



// WITH scale factor
__device__ __forceinline__ void mma_m16n8k64_SF_ptx(float &d0, float &d1, float &d2, float &d3, uint32_t const &a0,
    uint32_t const &a1, uint32_t const &a2, uint32_t const &a3,
    uint32_t const &b0, uint32_t const &b1, float const &c0, float const &c1,
    float const &c2, float const &c3, uint32_t const &sfa0,
    uint32_t const &sfb0,uint32_t const &tidA, uint32_t const &tidB,
    uint32_t const &bidA, uint32_t const &bidB) {
        // static constexpr uint16_t tidA = 0;
        // static constexpr uint16_t bidA = 0;
        // static constexpr uint16_t tidB = 0;
        // static constexpr uint16_t bidB = 0;
       
        asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X."
            "m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
            "{%0,  %1,  %2,  %3},"
            "{%4,  %5,  %6,  %7},"
            "{%8,  %9},"
            "{%10, %11, %12, %13},"
            "{%14},"
            "{%15, %16},"
            "{%17},"
            "{%18, %19};\n"
            : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), 
              "r"(b0), "r"(b1),
              "f"(c0), "f"(c1), "f"(c2), "f"(c3), 
              "r"(uint32_t(sfa0)),
              "h"((uint16_t)bidA), "h"((uint16_t)tidA), 
              "r"(uint32_t(sfb0)), 
              "h"((uint16_t)bidB), "h"((uint16_t)tidB));
    }

__global__ void mma_m16n8k64_SF_verify_kernel(float*D,uint32_t *A, uint32_t *B, float *C, uint32_t *SF_A,uint32_t *SF_B, uint16_t tidA, uint16_t tidB, int printLaneid = 0) {
    const int laneid = threadIdx.x % WARP_SIZE;
    const int index_a0 = laneid * 4;
    const int index_b0 = laneid * 2;
    const int index_c0 = laneid * 4;
    const int index_d0 = laneid * 4;
    const int index_sfa0 = laneid;
    const int index_sfb0 = laneid;
    mma_m16n8k64_SF_ptx(D[index_d0], D[index_d0+1], D[index_d0+2], D[index_d0+3], 
                         A[index_a0], A[index_a0+1], A[index_a0+2], A[index_a0+3], 
                         B[index_b0], B[index_b0+1], 
                         C[index_c0], C[index_c0+1], C[index_c0+2], C[index_c0+3],
                         SF_A[index_sfa0], SF_B[index_sfb0],tidA,tidB,0,0); 
    // printf("Laneid %d: A[0] = %08x, A[1] = %08x, A[2] = %08x, A[3] = %08x\n",
    //        laneid, A[index_a0], A[index_a0+1], A[index_a0+2], A[index_a0+3]);
    // printf("Laneid %d: B[0] = %08x, B[1] = %08x\n",
    //        laneid, B[index_b0], B[index_b0+1]);
    // printf("Laneid %d: C[0] = %8.3f, C[1] = %8.3f, C[2] = %8.3f, C[3] = %8.3f\n",
    //        laneid, C[index_c0], C[index_c0+1], C[index_c0+2], C[index_c0+3]);
    // printf("Laneid %d: D[0] = %8.3f, D[1] = %8.3f, D[2] = %8.3f, D[3] = %8.3f\n",
    //        laneid, D[index_d0], D[index_d0+1], D[index_d0+2], D[index_d0+3]);
    // printf("Laneid %d: SF_A = %08x, SF_B = %08x\n",
    //        laneid, SF_A[index_sfa0], SF_B[index_sfb0]);
    // printf("Laneid %d: tidA = %d, tidB = %d\n",laneid, (int)tidA, (int)tidB);
}

__global__ void mma_m16n8k64_SF_kernel(float*D,fp4 *input_A, fp4 *input_B, fp8 *sf_A,fp8 *sf_B, int printLaneid = 0) {
    const int laneid = threadIdx.x % WARP_SIZE;
    __shared__ fp4 A_sm[AM * AK];
    __shared__ fp4 B_sm[AK * AN];

    const int size_A = AM * AK / WARP_SIZE;
#pragma unroll
    for(int i = 0;i<size_A;i++){
        A_sm[laneid * size_A + i] = input_A[laneid * size_A + i];
    }
    const int size_B = AK * AN / WARP_SIZE;
#pragma unroll
    for(int i = 0;i<size_B;i++){
        int index = laneid * size_B + i;
        int row = index / AN;
        int col = index % AN;
        B_sm[col * AK + row] = input_B[row * AN + col];
    }
    __syncthreads();
    
    // load A
    int index_a0 = (laneid / 4) * AK + (laneid % 4) * 8;
    packed_a_t packA;
    if(laneid == printLaneid) printf("Packing A \n");
    packA.x = pack_fp4(&(A_sm[index_a0]),printLaneid);
    packA.y = pack_fp4(&(A_sm[index_a0 + (8 * AK)]),printLaneid);
    packA.z = pack_fp4(&(A_sm[index_a0 + 32]),printLaneid);
    packA.w = pack_fp4(&(A_sm[index_a0 + (8 * AK) + 32]),printLaneid);
    if(laneid == printLaneid) {
        printf("packA.x: ");
        print_binary32(packA.x);
        printf("\n");
        printf("packA.y: ");
        print_binary32(packA.y);
        printf("\n");
        printf("packA.z: ");
        print_binary32(packA.z);
        printf("\n");
        printf("packA.w: ");
        print_binary32(packA.w);
        printf("\n");
    }
    // load B (读取转置后的矩阵)
    int index_b0 = (laneid / 4) * AK + (laneid % 4) * 8; 
    packed_b_t packB;
    if(laneid == printLaneid) printf("Packing B \n");
    packB.x = pack_fp4(&(B_sm[index_b0]),printLaneid);
    packB.y = pack_fp4(&(B_sm[index_b0 + 32]),printLaneid); 
    if(laneid == printLaneid) {
        printf("packB.x: ");
        print_binary32(packB.x);
        printf("\n");
        printf("packB.y: ");
        print_binary32(packB.y);
        printf("\n");
    }
    // initialize packed C
    packed_f32psum_t packC = {0.0f};
    // initialize packed D
    packed_f32psum_t out = {0.0f};
    // initialize scale factors
    uint32_t sfa0;
    uint32_t sfb0;
    // fp8 e4m3 十六进制表示,uint32_t包含4个fp8
    // 1    0x38
    // 2    0x40
    // 4    0x48
    // 5    0x4a
    // 10   0x52
    if(laneid == 0) {
        sfb0 = 0x38000000;// 1 = 0x38383838;
    }else{
        sfb0 = 0x38383838;
    }
    sfa0 = 0x38383838;
   

    mma_m16n8k64_SF_ptx(out.data[0], out.data[1], out.data[2],out.data[3] ,
        packA.x, packA.y, packA.z, packA.w,
        packB.x, packB.y,
        packC.data[0], packC.data[1], packC.data[2], packC.data[3],
        sfa0, sfb0,0,0,0,0);

    // 将结果写回全局内存
    int index_d0 = laneid / 4 * AN + laneid % 4 * 2;
    D[index_d0] = out.data[0];
    D[index_d0 + 1] = out.data[1];
    D[index_d0 + AN * 8] = out.data[2];
    D[index_d0 + 1 + AN * 8] = out.data[3];
    if(laneid == printLaneid) {
        printf("D out: %8.3f %8.3f %8.3f %8.3f\n",
            static_cast<float>(out.data[0]),
            static_cast<float>(out.data[1]),
            static_cast<float>(out.data[2]),
            static_cast<float>(out.data[3]));
    }
}
/*
    usage：
        ./mma_fp4_block_scale_analysis      打印warp中0号线程的值
        ./mma_fp4_block_scale_analysis -1   不打印warp中对应编号的值
        ./mma_fp4_block_scale_analysis 8    打印warp中对应编号的值
*/
int main(int argc, char* argv[]){


    fp8 test_num_fp8 = fp8(10.0f);
    printf("test_num_fp8 = %f\n", static_cast<float>(test_num_fp8));
    print_binary_fp8(test_num_fp8.__x);
    
    fp4 test_num_fp4 = fp4(-2.0f);
    printf("test_num_fp4 = %f\n", static_cast<float>(test_num_fp4));
    print_binary_fp8(test_num_fp4.__x);

    int printLaneid =0;
    if(argc>1) printLaneid= atoi(argv[1]) % 32;
    fp4 fp4_range[16] {
        fp4(-6.0), fp4(-4.0), fp4(-3.0), fp4(-2.0),
        fp4(-1.5), fp4(-1.0), fp4(-0.5), fp4(-0.0), 
        fp4(+0.0), fp4(+0.5), fp4(+1.0), fp4(+1.5), 
        fp4(+2.0), fp4(+3.0), fp4(+4.0), fp4(+6.0),
    };
    // 打印所有fp4的值
    for(int i =0;i<16;i++){
        printf("------------------------------------\n");
        printf("i = %d\n",i);
        printf("fp4_range[%d] = %f\n", i, static_cast<float>(fp4_range[i]));
        print_binary_fp8(fp4_range[i].__x);
        printf("------------------------------------\n");
    }
    // 打印fp4x2的值
    float2 test_float2_num = float2{6.0f,-1.0f};
    double2 test_double2_num = double2{-6.0,1.0};
    fp4x2 test_fp4x2_num = fp4x2(test_float2_num);
    print_binary_fp8(test_fp4x2_num.__x);

    // 打印fp4x4的值
    float4 test_float4_num = float4{-6.0f,1.0f,6.0f,-1.0f};
    fp4x4 test_fp4x4_num = fp4x4(test_float4_num);
    print_binary16(test_fp4x4_num.__x);

    float float_range[16] {
        -6.0f, -4.0f, -3.0f, -2.0f,
        -1.5f, -1.0f, -0.5f, 0.0f,
        0.0f, 0.5f, 1.0f, 1.5f,
        2.0f, 3.0f, 4.0f, 6.0f
    };
    printf("-----------------------\n printLaneid = %d\n-------------------\n", printLaneid);


    fp4 host_a[AM*AK];
    fp4 host_b[AK*AN];
    float host_d[AM*AN] = {0.0f};
    // for(int i = 0; i < AM * AK; ++i) host_a[i] = fp4_range[i % 16];
    // for(int i = 0; i < AK * AN; ++i) host_b[i] = fp4_range[i % 16];
    
    for(int i = 0; i < AM * AK; ++i) host_a[i] = fp4(4.0);
    for(int i = 0; i < AK * AN; ++i) host_b[i] = fp4(4.0);
    bool is_test_sfA = false;
    if(is_test_sfA){
        for(int i = 0;i < 64;i++){
            if(i<16){
                host_a[i] = fp4(0.5f);
            }else if(16<=i && i<32){
                host_a[i] = fp4(1.0f);
            }else if(32<=i && i<48){
                host_a[i] = fp4(1.5f);
            }else if(48<=i && i<64){
                host_a[i] = fp4(2.0f);
            }
        }
    }else{// 测试sfb重新初始化B矩阵
        
        for(int i = 0;i < 64;i++){
            int index = i * 8;
            if(i<16){
                host_b[index] = fp4(0.5f);
            }else if(16<=i && i<32){
                host_b[index] = fp4(1.0f);
            }else if(32<=i && i<48){
                host_b[index] = fp4(1.5f);
            }else if(48<=i && i<64){
                host_b[index] = fp4(2.0f);
            }
        }
    }
    fp4 *dev_a;
    fp4 *dev_b;
    float *dev_d;
    fp8 *dev_sf_a;
    fp8 *dev_sf_b;
    
    CHECK_CUDA(cudaMalloc(&dev_a,sizeof(fp4) * AM * AK));
    CHECK_CUDA(cudaMalloc(&dev_b,sizeof(fp4) * AK * AN));
    CHECK_CUDA(cudaMalloc(&dev_d,sizeof(float) * AM * AN));
    CHECK_CUDA(cudaMalloc(&dev_sf_a,sizeof(fp8) * AM * AK));
    CHECK_CUDA(cudaMalloc(&dev_sf_b,sizeof(fp8) * AK * AN));
    
    CHECK_CUDA(cudaMemcpy(dev_a,host_a,sizeof(fp4)*AM*AK,cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dev_b,host_b,sizeof(fp4)*AK*AN,cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dev_d,host_d,sizeof(float)*AM*AN,cudaMemcpyHostToDevice));
    
    
    
    dim3 grid(1);
    dim3 block(32);

    mma_m16n8k64_SF_kernel<<<grid,block>>>(dev_d, dev_a, dev_b, dev_sf_a, dev_sf_b, printLaneid);

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(host_d,dev_d,sizeof(float)*AM*AN,cudaMemcpyDeviceToHost));
    
    // CPU端结果
    float host_a_float[AM*AK];
    float host_b_float[AK*AN];
    float host_d_float[AM*AN] = {0.0f};
    for(int i = 0; i < AM * AK; ++i) host_a_float[i] = static_cast<float>(host_a[i]);
    for(int i = 0; i < AK * AN; ++i) host_b_float[i] = static_cast<float>(host_b[i]);
    
    for(int i = 0; i < AM; ++i) {
        for(int j = 0; j < AN; ++j) {
            for(int k = 0; k < AK; ++k) {
                host_d_float[i * AN + j] += host_a_float[i * AK + k] * host_b_float[k * AN + j];
            }
        }
    }

    #define D_CPU
    #define D
    // #define B
    // #define A

    #ifdef D_CPU
    dumpEx1<float>(host_d_float, AM, AN, "D_CPU", 8, 2);
    #endif
    #ifdef A
    dumpEx1<fp4>(host_a,AM,AK, "A",8,8);
    #endif
    #ifdef B
    dumpEx1<fp4>(host_b,AK,AN, "B",8,1);
    #endif
    #ifdef D
    dumpEx1<float>(host_d,AM,AN, "D",8,2);
    #endif
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_d);
    cudaFree(dev_sf_a);
    cudaFree(dev_sf_b);
 
    // 从nunchaku 导出数据验证
    namespace verify_group_data = m16n8k64_data::group_tida0_tidb0;
    
    uint32_t *host_A0 = verify_group_data::A0;
    uint32_t *host_B0 = verify_group_data::B0;
    float *host_C0 = verify_group_data::C0;
    float *host_D0 = new float[32*4];
    uint32_t *host_SF_A0 = verify_group_data::SF_A0;
    uint32_t *host_SF_B0 = verify_group_data::SF_B0;
    uint16_t host_tidA0 = verify_group_data::tidA0;
    uint16_t host_tidB0 = verify_group_data::tidB0;
    

    uint32_t *dev_A0;
    uint32_t *dev_B0;
    float *dev_C0;
    float *dev_D0;
    uint32_t *dev_SF_A0;
    uint32_t *dev_SF_B0;

    CHECK_CUDA(cudaMalloc(&dev_A0, sizeof(uint32_t) * 32 * 4));
    CHECK_CUDA(cudaMalloc(&dev_B0, sizeof(uint32_t) * 32 * 2));
    CHECK_CUDA(cudaMalloc(&dev_C0, sizeof(float) * 32 * 4));
    CHECK_CUDA(cudaMalloc(&dev_D0, sizeof(float) * 32 * 4));
    CHECK_CUDA(cudaMalloc(&dev_SF_A0, sizeof(uint32_t) * 32));
    CHECK_CUDA(cudaMalloc(&dev_SF_B0, sizeof(uint32_t) * 32));


    CHECK_CUDA(cudaMemcpy(dev_A0, host_A0, sizeof(uint32_t) * 32 * 4, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dev_B0, host_B0, sizeof(uint32_t) * 32 * 2, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dev_C0, host_C0, sizeof(float) * 32 * 4, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dev_SF_A0, host_SF_A0, sizeof(uint32_t) * 32, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dev_SF_B0, host_SF_B0, sizeof(uint32_t) * 32, cudaMemcpyHostToDevice));

    mma_m16n8k64_SF_verify_kernel<<<1, 32>>>(dev_D0, dev_A0, dev_B0, dev_C0, dev_SF_A0, dev_SF_B0, host_tidA0, host_tidB0, printLaneid);

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(host_D0, dev_D0, sizeof(float) * 32 * 4, cudaMemcpyDeviceToHost));

    // dumpEx1<float>(host_D0, 32, 4, "D0_calcuate", 4, 1);
    // dumpEx1<float>(verify_group_data::D0, 32, 4, "D0_row", 4, 1);
    #if 0
    // 验证不同8位值的数量
    std::set<uint8_t> unique_bytes;

    for (uint32_t val : verify_group_data::SF_B0) {
        unique_bytes.insert(val & 0xFF);         // 字节0
        unique_bytes.insert((val >> 8) & 0xFF);  // 字节1
        unique_bytes.insert((val >> 16) & 0xFF); // 字节2
        unique_bytes.insert((val >> 24) & 0xFF); // 字节3
    }

    std::cout << "不同8位值的总数: " << unique_bytes.size() << std::endl;
    std::cout << "不同8位值: ";
    for (uint8_t byte : unique_bytes) {
        printf("%02x ", byte);

        
    }
    std::cout << std::endl;
    #endif
}