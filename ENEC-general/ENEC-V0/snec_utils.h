/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
 * This file constains code of cpu debug and npu code.We read data from bin file
 * and write result to file.
 */

#ifndef snec_UTILS_H
#define snec_UTILS_H

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
constexpr uint32_t BUFFER_NUM = 1;
constexpr uint32_t BLOCK_NUM = 48;
constexpr uint32_t HISTOGRAM_BINS = 256;
constexpr uint32_t DATA_BLOCK_BYTE_NUM_H = 16 * 4096;
constexpr uint32_t DATA_BLOCK_ELEMENT_NUM_C = 8192;
struct Header {
    uint32_t dataBlockSize; 
    uint32_t dataBlockNum;
    uint16_t threadBlockNum;
    uint16_t compLevel; 
    uint32_t totalUncompressedBytes;
    uint32_t totalCompressedBytes;
    uint16_t tileLength;  
    uint16_t dataType;  
    uint16_t mblLength;  
    uint16_t options;  
    uint32_t histogramBytes; 
};
inline uint8_t *getTable(Header *cphd, uint8_t *compressed) {
    return compressed + 32;
}
inline uint8_t *getMsdata(Header *cphd, uint8_t *compressed) {
    return getTable(cphd, compressed) + HISTOGRAM_BINS;
}
inline uint8_t *getMbl(Header *cphd, uint8_t *compressed) {
    return getMsdata(cphd, compressed) + (cphd->totalUncompressedBytes + 2 - 1) / 2;
}
inline uint8_t *getCompSizePrefix(Header *cphd, uint8_t *compressed) {
    if (cphd->dataType == 0 | cphd->dataType == 1)
        return getMbl(cphd, compressed) + cphd->dataBlockNum * (cphd->dataBlockSize / (cphd->tileLength * sizeof(uint16_t)) / 2);
    else if (cphd->dataType == 2)
        return getMbl(cphd, compressed) + cphd->dataBlockNum * (cphd->dataBlockSize / (cphd->tileLength * sizeof(float)) / 2);
}
inline uint8_t *getCompressed_exp(Header *cphd, uint8_t *compressed) {
    return getCompSizePrefix(cphd, compressed) + cphd->threadBlockNum * sizeof(uint32_t);
}
inline int getFinalbufferSize(uint32_t byteSize, uint32_t tileNum, uint32_t DATA_BLOCK_BYTE_NUM_C) {
    int datablockNum = (byteSize + DATA_BLOCK_BYTE_NUM_C - 1) / DATA_BLOCK_BYTE_NUM_C;
    int datablockNumPerBLOCK = (datablockNum + BLOCK_NUM - 1) / BLOCK_NUM;
    int FinalBufferSize = 32 + HISTOGRAM_BINS * sizeof(uint8_t) + DATA_BLOCK_BYTE_NUM_C / 2 * datablockNum + tileNum / 2 * datablockNum + BLOCK_NUM * 4 + (DATA_BLOCK_BYTE_NUM_C / 2 * datablockNumPerBLOCK) * BLOCK_NUM;
    return FinalBufferSize;
}
inline float computeCr(uint32_t inputByteSize, uint32_t compressedSize) {
    if (compressedSize == 0)
        return 0.0f;
    return static_cast<float>(inputByteSize) / static_cast<float>(compressedSize);
}
#endif