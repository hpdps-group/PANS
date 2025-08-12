# ENEC(Efficient NPU Entropy Compressor)

(C) 2025 by Institute of Computing Technology, Chinese Academy of Sciences. 
- Developer: Jinwu Yang 
- Advisor: Dingwen Tao, Guangming Tan

## 格式转换文件
pt_to_dat.py: 一个将.pt文件转.dat文件的python程序，小端处理

## General版本

V0: 初始版本（输入数据必须为16B的整数倍）

V1: 在V0基础上，全部操作都在Device侧完成，同时解决了解压结果出现随机的问题（UB重新分配与规划）

V2: 在V1基础上，删减了冗余代码

V3: 在V2基础上，横相邻16元素最大位宽->横分散16元素最大位宽

V4: 在V3基础上，对解压缩的前缀和计算进行了优化

V5: 在V4基础上，对取数掩码等等计算进行优化，同时为了速度直接将bf16切成两字节

V6: 在V5基础上，对压缩内核中gather提取低16比特与解压中反向gather进行优化（提取32位->提取16位）

V7: 在V6基础上，对压缩符号编号替换的gather进行线性拟合优化，同时回退到全部提取指数的操作

V8: 在V6基础上，压缩每两轮一次gather输出，解压每两轮一次读入，同时缓冲区更新变成减法操作（输入数据此时必须为32KB的整数倍）

V9: 在V8基础上，对压缩和解压缩的table符号替换的gather进行优化（gather的数据量减半，时间减小），kernel内分成编码替换与bit-packing两块独立部分

V10: 在V9基础上，修改压缩文件内容（消除cmbl，保存对应的比较掩码），尽可能提升解压缩吞吐量

## Quantized版本

V0: 双量化比特值

V1: 三量化比特值
