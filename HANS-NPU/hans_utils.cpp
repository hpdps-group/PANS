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

template <typename U, typename V>
constexpr __host__ __device__ auto divDown(U a, V b) -> decltype(a + b) {
  return (a / b);
}

template <typename U, typename V>
constexpr __host__ __device__ auto divUp(U a, V b) -> decltype(a + b) {
  return (a + b - 1) / b;
}

template <typename U, typename V>
constexpr __host__ __device__ auto roundDown(U a, V b) -> decltype(a + b) {
  return divDown(a, b) * b;
}

template <typename U, typename V>
constexpr __host__ __device__ auto roundUp(U a, V b) -> decltype(a + b) {
  return divUp(a, b) * b;
}

template <typename T>
constexpr __host__ __device__ bool isEvenDivisor(T a, T b) {
  return (a % b == 0) && ((a / b) >= 1);
}

template <class T>
constexpr __host__ __device__ T pow(T n, T power) {
  return (power > 0 ? n * pow(n, power - 1) : 1);
}

template <class T>
constexpr __host__ __device__ T pow2(T n) {
  return pow(2, (T)n);
}

static_assert(pow2(8) == 256, "pow2");

template <typename T>
constexpr __host__ __device__ int log2(T n, int p = 0) {
  return (n <= 1) ? p : log2(n / 2, p + 1);
}

static_assert(log2(2) == 1, "log2");
static_assert(log2(3) == 1, "log2");
static_assert(log2(4) == 2, "log2");

template <typename T>
constexpr __host__ __device__ bool isPowerOf2(T v) {
  return (v && !(v & (v - 1)));
}

static_assert(isPowerOf2(2048), "isPowerOf2");
static_assert(!isPowerOf2(3333), "isPowerOf2");

template <typename T>
constexpr __host__ __device__ T nextHighestPowerOf2(T v) {
  return (isPowerOf2(v) ? (T)2 * v : ((T)1 << (log2(v) + 1)));
}

static_assert(nextHighestPowerOf2(1) == 2, "nextHighestPowerOf2");
static_assert(nextHighestPowerOf2(2) == 4, "nextHighestPowerOf2");
static_assert(nextHighestPowerOf2(3) == 4, "nextHighestPowerOf2");
static_assert(nextHighestPowerOf2(4) == 8, "nextHighestPowerOf2");

static_assert(nextHighestPowerOf2(15) == 16, "nextHighestPowerOf2");
static_assert(nextHighestPowerOf2(16) == 32, "nextHighestPowerOf2");
static_assert(nextHighestPowerOf2(17) == 32, "nextHighestPowerOf2");

static_assert(
    nextHighestPowerOf2(1536000000u) == 2147483648u,
    "nextHighestPowerOf2");
static_assert(
    nextHighestPowerOf2((size_t)2147483648ULL) == (size_t)4294967296ULL,
    "nextHighestPowerOf2");

template <typename T>
constexpr __host__ __device__ T nextLowestPowerOf2(T v) {
  return (isPowerOf2(v) ? v / (T)2 : ((T)1 << (log2(v))));
}

static_assert(nextLowestPowerOf2(1) == 0, "nextLowestPowerOf2");
static_assert(nextLowestPowerOf2(2) == 1, "nextLowestPowerOf2");
static_assert(nextLowestPowerOf2(3) == 2, "nextLowestPowerOf2");
static_assert(nextLowestPowerOf2(4) == 2, "nextLowestPowerOf2");

static_assert(nextLowestPowerOf2(15) == 8, "nextLowestPowerOf2");
static_assert(nextLowestPowerOf2(16) == 8, "nextLowestPowerOf2");
static_assert(nextLowestPowerOf2(17) == 16, "nextLowestPowerOf2");

inline __host__ __device__ bool isPointerAligned(const void* p, int align) {
  return reinterpret_cast<uintptr_t>(p) % align == 0;
}

// Returns the increment needed to aligned the pointer to the next highest
// aligned address
template <int Align>
inline __host__ __device__ uint32_t getAlignmentRoundUp(const void* p) {
  static_assert(isPowerOf2(Align));
  uint32_t diff = uint32_t(uintptr_t(p) & uintptr_t(Align - 1));
  return diff == 0 ? 0 : uint32_t(Align) - diff;
}

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