
/**
 * @file hello_world.cpp
 *
 * Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
// hans_utils.h
#ifndef HANS_UTILS_H
#define HANS_UTILS_H

#include <fcntl.h>
#include <sys/stat.h>
#include <cassert>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include <unistd.h>
#include <chrono>
#include <bitset>

constexpr uint32_t BUFFER_NUM = 1; // 双缓冲
constexpr uint32_t BLOCK_NUM = 48;// block的数量, 必须是8的倍数
constexpr uint32_t HISTOGRAM_BINS = 256;// 尽可能是2的幂，直方图桶数
constexpr uint32_t DATA_BLOCK_BYTE_NUM_C = 16384;
//8 * 2048;// 单位为字节

struct Header{
    uint32_t dataBlockSize; // 数据块大小, 字节为单位
    uint32_t dataBlockNum; // 数据块数量
    uint16_t threadBlockNum; // 线程块数量，高16位存储线程块数量，低16位存储压缩级别
    uint16_t compLevel; // 压缩级别
    uint32_t totalUncompressedBytes; // 总未压缩字节数
    uint32_t totalCompressedBytes; // 总压缩字节数  
    uint16_t tileLength; // Tile长度
    uint16_t dataType; // 数据类型，0代表bf16，1代表fp16，2代表fp32
    uint16_t mblLength; // mbl长度
    uint16_t options; // 选项，0代表CPU，1代表NV_GPU，2代表AMD_GPU，3代表NPU
    uint32_t HistogramBytes; // 直方图保存字节数
};

inline uint8_t* getTable(Header *cphd, uint8_t *compressed)
{
    return compressed + 32;
}

inline uint8_t* getMsdata(Header *cphd, uint8_t *compressed)
{
    return getTable(cphd, compressed) + HISTOGRAM_BINS;
}

inline uint8_t* getMbl(Header *cphd, uint8_t *compressed)
{
    return getMsdata(cphd, compressed) + (cphd->totalUncompressedBytes + 2 - 1) / 2;
}

inline uint8_t* getCompSizePrefix(Header *cphd, uint8_t *compressed)
{
    if (cphd->dataType == 0 | cphd->dataType == 1)
        return getMbl(cphd, compressed) + cphd->dataBlockNum * (cphd->dataBlockSize / (cphd->tileLength * sizeof(uint16_t)) / 2);
    else if (cphd->dataType == 2)
        return getMbl(cphd, compressed) + cphd->dataBlockNum * (cphd->dataBlockSize / (cphd->tileLength * sizeof(float)) / 2);
}

inline uint8_t* getCompressed_exp(Header *cphd, uint8_t *compressed)
{
    return getCompSizePrefix(cphd, compressed) + cphd->threadBlockNum * sizeof(uint32_t);
}

inline int getFinalbufferSize(uint32_t byteSize, uint32_t tileNum){
    // int FinalBufferSize = 0;
    int datablockNum = (byteSize + DATA_BLOCK_BYTE_NUM_C - 1) / DATA_BLOCK_BYTE_NUM_C;
    int datablockNumPerBLOCK = (datablockNum + BLOCK_NUM - 1) / BLOCK_NUM;
    int FinalBufferSize =   32 + 
                            HISTOGRAM_BINS * sizeof(uint8_t) +   
                            DATA_BLOCK_BYTE_NUM_C / 2 * datablockNum + 
                            tileNum / 2 * datablockNum + 
                            BLOCK_NUM * 4 + // 每个block的压缩大小前缀和
                            (DATA_BLOCK_BYTE_NUM_C / 2 * datablockNumPerBLOCK) * BLOCK_NUM;
    return FinalBufferSize;
}

inline float computeCr(uint32_t inputByteSize, uint32_t compressedSize)
{
    if (compressedSize == 0) {
        return 0.0f; // 避免除以零
    }
    return static_cast<float>(inputByteSize) / static_cast<float>(compressedSize);
}
#endif // HANS_UTILS_H