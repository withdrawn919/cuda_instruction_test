#include "../dequantize_fp4_fp8/wmma_utils.cuh"
#include "../dequantize_fp4_fp8/wmma.cuh"


__global__ void ldmatrix_test_kernel(fp8* input_A,fp8* output_A,int printLaneid = 0) {
    const size_t laneid = threadIdx.x % WARP_SIZE;
    __shared__ fp8 input_A_smem[16 * 16];
    __shared__ fp8 output_A_smem[16 * 16];
    
    int size_A = 16 * 16 / WARP_SIZE;
#pragma unroll
    for(int i = 0; i < size_A; i++){
        input_A_smem[threadIdx.x * size_A + i] = input_A[threadIdx.x * size_A + i];
    }
    __syncthreads();
    int aTile_index = laneid % 16 * 16 + laneid / 16 * 8;
    uint32_t dst0;
    uint32_t dst1;
    uint32_t smem_ptr = __cvta_generic_to_shared(&input_A_smem[aTile_index]);
    asm volatile ("ldmatrix.sync.aligned.m16n16.x1.trans.shared.b8 {%0, %1}, [%2];\n"
        : "=r"(dst0), "=r"(dst1)
        :  "r"(smem_ptr));
    if(laneid == printLaneid) {
        print_binary32(dst0);
        print_binary32(dst1);
        fp8 r0_0;
        r0_0.__x =static_cast<__nv_fp8_storage_t>((dst0>>24) & 0xff);
        fp8 r0_1;
        r0_1.__x = static_cast<__nv_fp8_storage_t>((dst0>>16) & 0xff);
        fp8 r0_2;
        r0_2.__x = static_cast<__nv_fp8_storage_t>((dst0>>8) & 0xff);
        fp8 r0_3;
        r0_3.__x = static_cast<__nv_fp8_storage_t>(dst0 & 0xff);
        fp8 r1_0;
        r1_0.__x = static_cast<__nv_fp8_storage_t>((dst1>>24) & 0xff);
        fp8 r1_1;
        r1_1.__x = static_cast<__nv_fp8_storage_t>((dst1>>16) & 0xff);
        fp8 r1_2;
        r1_2.__x = static_cast<__nv_fp8_storage_t>((dst1>>8) & 0xff);
        fp8 r1_3;
        r1_3.__x = static_cast<__nv_fp8_storage_t>(dst1 & 0xff);
        print_binary_fp8(r0_0.__x);
        print_binary_fp8(r0_1.__x);
        print_binary_fp8(r0_2.__x);
        print_binary_fp8(r0_3.__x);
        print_binary_fp8(r1_0.__x);
        print_binary_fp8(r1_1.__x);
        print_binary_fp8(r1_2.__x);
        print_binary_fp8(r1_3.__x); 
        printf("r0_0: %f, r0_1: %f, r0_2: %f, r0_3: %f\n",
               static_cast<float>(r0_0), static_cast<float>(r0_1),
               static_cast<float>(r0_2), static_cast<float>(r0_3));
        printf("r1_0: %f, r1_1: %f, r1_2: %f, r1_3: %f\n",
               static_cast<float>(r1_0), static_cast<float>(r1_1),
               static_cast<float>(r1_2), static_cast<float>(r1_3));
    }
}

int main(int argc,char* argv[]) {
    int printLaneid=0;
    if(argc > 1) {
        printLaneid = atoi(argv[1]) % 32;
        printf("printLaneid: %d\n", printLaneid);
    } 
    fp8 host_A_input[16*16];
    fp8 host_A_output[16*16] = {fp8(0.0f)};
    fp8 *dev_A_input;
    fp8 *dev_A_output;

    for(int i = 0; i < 16 * 16; i++) {
        host_A_input[i] = fp8(static_cast<float>(i));
    }
    dumpEx1<fp8>(host_A_input, 16, 16,"host_A_input",8,4);
    cudaMalloc((void**)&dev_A_input, sizeof(fp8) * 16 * 16);
    cudaMalloc((void**)&dev_A_output, sizeof(fp8) * 16 * 16);
    cudaMemcpy(dev_A_input, host_A_input, sizeof(fp8) * 16 * 16, cudaMemcpyHostToDevice);
    ldmatrix_test_kernel<<<1, 32>>>(dev_A_input, dev_A_output, printLaneid);
    cudaDeviceSynchronize();
    return 0;
}