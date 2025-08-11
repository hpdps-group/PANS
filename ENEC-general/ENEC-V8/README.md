## 自研NPU熵压缩器-SNEC 说明

## 安装方法
mkdir build && cd build && cmake .. && make

## 使用方法

压缩：
./compress sourcefile tempfile inputBytesNum

解压缩：
./decompress tempfile resfile sourcefile 

注意：解压缩自动进行正确性验证
