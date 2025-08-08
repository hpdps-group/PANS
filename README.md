# ENEC(Efficient NPU Entropy Compressor)

(C) 2025 by Institute of Computing Technology, Chinese Academy of Sciences. 
- Developer: Jinwu Yang 
- Advisor: Dingwen Tao, Guangming Tan

## 格式转换文件
pt_to_dat.py: 一个将.pt文件转.dat文件的python程序，小端处理

## General版本

V0: 初始版本

V1: 在V0基础上，全部操作都在Device侧完成，同时解决了解压结果出现随机的问题（UB重新分配与规划）

V2: 在V1基础上，删减了冗余代码

V3: 在V2基础上，横相邻16元素最大位宽->横分散16元素最大位宽

V4: 在V3基础上，对解压缩的前缀和计算进行了优化

V5: 在V4基础上，对取数掩码等等计算进行优化，同时为了速度直接将bf16切成两字节

V6: 在V5基础上，进一步增大数据块大小，数据块由16KB增大到32KB

## Quantized版本

V0: 双量化比特值

V1: 三量化比特值
