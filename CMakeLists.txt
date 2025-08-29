cmake_minimum_required(VERSION 3.10.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)  
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

project(fp4_fp8 VERSION 1.0.0 LANGUAGES CUDA CXX C)

set(CMAKE_BUILD_TYPE Debug)

set(CMAKE_CUDA_ARCHITECTURES 120a)  

find_package(CUDAToolkit)
enable_language(CXX)
enable_language(CUDA)


add_executable(mma_fp4 ./mma_fp4_fp8/mma_fp4.cu)
# target_compile_options(mma_fp4 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -keep>)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  target_compile_options(mma_fp4 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:$<$<CONFIG:Debug>:-O0 -G -g -fno-omit-frame-pointer>>)
endif()


add_executable(mma_fp8 ./mma_fp4_fp8/mma_fp8.cu)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  target_compile_options(mma_fp8 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:$<$<CONFIG:Debug>:-O0 -G -g -fno-omit-frame-pointer>>)
endif()

add_executable(mma_fp4_sf ./mma_fp4_fp8/mma_fp4_sf.cu)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  target_compile_options(mma_fp4_sf PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:$<$<CONFIG:Debug>:-O0 -G -g -fno-omit-frame-pointer>>)
endif()

add_executable(mma_fp8_sf ./mma_fp4_fp8/mma_fp8_sf.cu)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  target_compile_options(mma_fp8_sf PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:$<$<CONFIG:Debug>:-O0 -G -g -fno-omit-frame-pointer>>)
endif()


add_executable(dequantize_fp4_fp82float ./dequantize_fp4_fp8/dequantize_fp4_fp82float.cu)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  target_compile_options(dequantize_fp4_fp82float PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:$<$<CONFIG:Debug>:-O0 -G -g -fno-omit-frame-pointer>>)
endif()

add_executable(dequantize_fp8 ./dequantize_fp4_fp8/dequantize_fp8.cu)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  target_compile_options(dequantize_fp8 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:$<$<CONFIG:Debug>:-O0 -G -g -fno-omit-frame-pointer>>)
endif()


add_executable(cvt_fp8_hard cvt/cvt_fp8_hard.cu)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  target_compile_options(cvt_fp8_hard PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:$<$<CONFIG:Debug>:-O0 -G -g -fno-omit-frame-pointer>>)
endif()

add_executable(cvt_fp8_soft cvt/cvt_fp8_soft.cu)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  target_compile_options(cvt_fp8_soft PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:$<$<CONFIG:Debug>:-O0 -G -g -fno-omit-frame-pointer>>)
endif()

add_executable(cvt_fp4_hard_soft cvt/cvt_fp4_hard_soft.cu)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  target_compile_options(cvt_fp4_hard_soft PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:$<$<CONFIG:Debug>:-O0 -G -g -fno-omit-frame-pointer>>)
endif()

add_executable(16x8x16x2_16x16x16 ./mma_fp4_fp8/16x8x16x2_16x16x16.cu)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  target_compile_options(16x8x16x2_16x16x16 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:$<$<CONFIG:Debug>:-O0 -G -g -fno-omit-frame-pointer>>)
endif()

add_executable(wmma_fp8_fp32_fp8_m16n8k16_m16n8k8 ./dequantize_fp4_fp8/wmma_fp8_fp32_fp8_m16n8k16_m16n8k8.cu)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  target_compile_options(wmma_fp8_fp32_fp8_m16n8k16_m16n8k8 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:$<$<CONFIG:Debug>:-O0 -G -g -fno-omit-frame-pointer>>)
endif()

add_executable(mma_fp4_block_scale_analysis ./mma_fp4_fp8/mma_fp4_block_scale_analysis.cu)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  target_compile_options(mma_fp4_block_scale_analysis PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:$<$<CONFIG:Debug>:-O0 -G -g>>)
endif()

add_executable(ldmatrix_m16n16 ldmatrix/ldmatrix_m16n16.cu)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  target_compile_options(ldmatrix_m16n16 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:$<$<CONFIG:Debug>:-O0 -G -g>>)
endif()

add_executable(mma_ldmatrix_col_row ./mma_fp4_fp8/mma_ldmatrix_col_row.cu)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  target_compile_options(mma_ldmatrix_col_row PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:$<$<CONFIG:Debug>:-O0 -G -g>>)
endif()

add_executable(cp_async_ca cp/cp_async_ca.cu)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  target_compile_options(cp_async_ca PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:$<$<CONFIG:Debug>:-O0 -G -g>>)
endif()

add_executable(m16n16k64 mma_fp4_fp8/m16n16k64.cu)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  target_compile_options(m16n16k64 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:$<$<CONFIG:Debug>:-O0 -G -g>>)
endif()

add_executable(m16n16k64_fused mma_fp4_fp8/m16n16k64_fused.cu)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  target_compile_options(m16n16k64_fused PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:$<$<CONFIG:Debug>:-O0 -G -g>>)
endif()

add_executable(m16n16k64dq_cuda_core dequantize_fp4_fp8/m16n16k64dq_cuda_core.cu)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  target_compile_options(m16n16k64dq_cuda_core PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:$<$<CONFIG:Debug>:-O0 -G -g>>)
endif()

add_executable(m16n8k32_s8_s32 mma/m16n8k32_s8_s32.cu)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  target_compile_options(m16n8k32_s8_s32 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:$<$<CONFIG:Debug>:-O0 -G -g>>)
endif()

add_executable(sp_m16n8k32_bf16_f32 mma_sp/sp_m16n8k32_bf16_f32.cu)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  target_compile_options(sp_m16n8k32_bf16_f32 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:$<$<CONFIG:Debug>:-O0 -G -g>>)
endif()

add_executable(ldmatrix_x1_x2_x4_analysis ldmatrix/ldmatrix_x1_x2_x4_analysis.cu)
# if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  target_compile_options(ldmatrix_x1_x2_x4_analysis PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:$<$<CONFIG:Debug>:-O0 -G -g>>)
# endif()

# set(CUTLASS_NVCC_ARCHS 120)
# message(STATUS "Using CUDA architectures: ${CUTLASS_NVCC_ARCHS}")
# set(CUTLASS_PATH "/home/jingjiawei/workspace/cutlass/include")
# set(CUTLASS_UTIL_PATH "/home/jingjiawei/workspace/cutlass/tools/util/include")
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
# set(CUTLASS_EXAMPLE_COMMON_PATH "/home/jingjiawei/workspace/cutlass/examples/common")

# add_executable(ldmatrix_x1_x2_x4_analysis ldmatrix/ldmatrix_x1_x2_x4_analysis.cu)
# target_link_libraries(ldmatrix_x1_x2_x4_analysis PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})
# target_include_directories(ldmatrix_x1_x2_x4_analysis PRIVATE ${CUTLASS_PATH})
# target_include_directories(ldmatrix_x1_x2_x4_analysis PRIVATE ${CUTLASS_EXAMPLE_COMMON_PATH})
# target_include_directories(ldmatrix_x1_x2_x4_analysis PRIVATE ${CUTLASS_UTIL_PATH})
# if(CMAKE_BUILD_TYPE STREQUAL "Debug")
#   target_compile_options(ldmatrix_x1_x2_x4_analysis PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:$<$<CONFIG:Debug>:-O0 -G -g>>)
# else()
#   target_compile_options(ldmatrix_x1_x2_x4_analysis PRIVATE -lineinfo)
# endif()