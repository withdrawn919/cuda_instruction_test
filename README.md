# FP4/FP8 Samples

这个项目中的所有代码均在5060ti上测试通过。	

- mma_fp4_fp8

  - mma_fp4.cu：封装了不带sf功能的fp4 mma的指令

  - mma_fp4_sf.cu：封装了带sf功能的fp4 mma的指令

  - mma_fp8.cu：封装了不带sf功能的fp8 mma的指令

  - mma_fp8_sf.cu：封装了带sf功能的fp8 mma的指令

  - mma_fp4_block_scale_analysis.cu：fp4的带有block_scale的mma指令行为分析，封装了带SF和不带SF的mma指令。运行程序可以加参数，-1表示不打印线程的值，0-31表示打印对应的线程值。
    main函数中有打印fp8和fp4的代码，罗列了所有fp4能表示的数。

  - mma_ldmatirix_col_row.cu：使用封装的wmma库测试行主序和列主序对计算结果的影响

  - 16x8x16x2_16x16x16.cu：测试两个小矩阵乘法拼成一个大矩阵，其中用到了cp、ldmatirx、mma PTX指令，还用到了封装的wmma库计算gemm

- dequantize_fp4_fp8

  - dequantize_fp4_fp82float.cu：基于fp8反量化的方式，将量化后的fp4矩阵做完gemm的fp32结果，分块**点乘**fp8量化因子.

    ​	A矩阵：dtype=fp32，	shape=[8,16]

    ​	B矩阵：dtype=fp8，	  shape=[8,1]

    ​	D矩阵：dtype=fp32，	shape=[8,16]

  - dequantize_fp8.cu：实现fp8的反量化，整个fp32矩阵**点乘**以相同的量化因子

    ​	A矩阵：dtype=fp8，	   shape=[8,16]

    ​	β标量：dtype=fp32

    ​	D矩阵：dtype=fp32，	shape=[8,16]

  - wmma_m16n8k16.cu：封装了m16n8k16的tensor core指令

  - wmma_m16n8k16.cu：封装了m16n8k8的tensor core指令

  - wmma_fp8_fp32_fp8_m16n8k16_m16n8k8.cu: 将m16n8k16与m16n8k8两个tensor core融合到一个kernel中，模拟软件方法做fp4反量化

  - mma_m16n16k64_cudacore.cu：使用cuda core在计算过程中反量化

  - m16n16k64dq_cuda_core.cu：cuda core反量化测试样例

- cvt
  - cvt_fp8_hard.cu：测试float2fp8的硬件转换

  - cvt_fp8_soft.cu：测试float2fp8的软件转换

  - cvt_fp4_hard_soft.cu：测试float2fp4的硬件和软件转换

- ldmatrix（shared->register）

  - ldmatrix_m16n16.cu：测试m16n16加载8bit位宽数据以及trans

- cp（global->shared）

  - cp_async_ca.cu：cp指令行为分析，

- nsight-report
  
    这个路径下保存程序的分析报告



