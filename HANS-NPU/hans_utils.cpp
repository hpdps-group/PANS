/**
 * @file hello_world.cpp
 *
 * Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cassert>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

constexpr int32_t BLOCK_NUM = 256;

#define CHECK_ACL(x)                                                                        \
    do {                                                                                    \
        aclError __ret = x;                                                                 \
        if (__ret != ACL_ERROR_NONE) {                                                      \
            std::cerr << __FILE__ << ":" << __LINE__ << " aclError:" << __ret << std::endl; \
        }                                                                                   \
    } while (0);

struct CoalescedHeader {
    static inline uint32_t getCompressedOverhead(uint32_t numBlocks) { //bits
        int kAlignment = kBlockAlignment / sizeof(uint2);
        if (kAlignment == 0) kAlignment = 1;

        return sizeof(CoalescedHeader) + //header
               sizeof(uint16_t) * kNumSymbols + // table
               divUp(3 * numBlocks, 8) + // max_bits_length
               sizeof(uint32_t) * roundUp(numBlocks, kAlignment); // compressed_size and compress_prefix
    }
    
    inline uint32_t getTotalCompressedSize() {
        return getCompressedOverhead()  +
               getTotalCompressedWords();
    }

    inline uint32_t getCompressedOverhead() {
        return getCompressedOverhead(getNumBlocks());
    }

    inline float getCompressionRatio() {
        return static_cast<float>(getTotalCompressedWords()) /
               (static_cast<float>(getTotalUncompressedWords()));
    }

    inline uint32_t getNumBlocks() { return numBlocks; }
    inline void setNumBlocks(uint32_t nb) { numBlocks = nb; }

    inline uint32_t getTotalUncompressedWords() { return totalUncompressedWords; }
    inline void setTotalUncompressedWords(uint32_t words) { totalUncompressedWords = words; }

    inline uint32_t getTotalCompressedWords() { return totalCompressedWords; }
    inline void setTotalCompressedWords(uint32_t words) { totalCompressedWords = words; }

    inline uint16_t* getSymbolTable() { return reinterpret_cast<uint16_t*>(this + 1); }

    inline uint8_t* getMaxbit_length() { return reinterpret_cast<uint8_t*>(getSymbolTable() + kNumSymbols); }

    inline uint2* getBlockWords(uint32_t numBlocks) {
        return reinterpret_cast<uint2*>(getMaxbit_length() + divUp(3 * numBlocks, 8));
    }

    inline uint8_t* getBlockDataStart(uint32_t numBlocks) {
        constexpr int kAlignment = kBlockAlignment / sizeof(uint2) == 0
        ? 1 : kBlockAlignment / sizeof(uint2);
        return reinterpret_cast<uint8_t*>(getBlockWords(numBlocks) + roundUp(numBlocks, kAlignment));
    }

    uint32_t numBlocks;
    uint32_t totalUncompressedWords;
    uint32_t totalCompressedWords;
    uint32_t options;// CPU , NV_GPU, AMD_GPU, NPU
    uint32_t unuse0;
    uint32_t unuse1;
    uint32_t unuse2;
    uint32_t unuse3;
};