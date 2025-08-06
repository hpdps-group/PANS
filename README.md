# ENEC(Efficient NPU Entropy Compressor)

(C) 2025 by Institute of Computing Technology, Chinese Academy of Sciences. 
- Developer: Jinwu Yang 
- Advisor: Dingwen Tao, Guangming Tan

## 格式转换文件
pt_to_dat.py: 一个将.pt文件转.dat文件的python程序，小端处理

## General版本

V0: 初始版本

V1: 在V0基础上，全部操作都在Device侧完成，同时解决了解压结果出现随机的问题（UB重新分配与规划）

V2: 在V1基础上删减了冗余代码

V3: 在V2基础上，横相邻16元素最大位宽->横分散16元素最大位宽

## Quantized版本

V0: 双量化比特值

V1: 三量化比特值
