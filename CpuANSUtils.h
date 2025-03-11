#pragma once

#include <cstdint>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <cassert>
#include <vector>
#include <cstring>
#include <omp.h>

namespace cpu_ans {

using ANSStateT = uint32_t;
using ANSEncodedT = uint16_t;
using ANSDecodedT = uint8_t;

struct ANSWarpState {
    ANSStateT warpState[32]; // kWarpSize is 32 in CUDA code
};

struct 
//alignas(8) 
uint2 { // 保持8字节对齐
    uint32_t x;
    uint32_t y;
};
struct uint4 { uint32_t x, y, z, w; };

constexpr int kWarpSize = 32;

uint32_t divDown(uint32_t a, uint32_t b) { return a / b; }
uint32_t getAlignmentRoundUp(size_t alignment, const void* ptr) {
    return (alignment - (reinterpret_cast<uintptr_t>(ptr) % alignment)) % alignment;
}

template <typename U, typename V>
auto divDown(U a, V b) -> decltype(a + b) {
  return (a / b);
}

template <typename U, typename V>
auto divUp(U a, V b) -> decltype(a + b) {
  return (a + b - 1) / b;
}

template <typename U, typename V>
auto roundDown(U a, V b) -> decltype(a + b) {
  return divDown(a, b) * b;
}

template <typename U, typename V>
auto roundUp(U a, V b) -> decltype(a + b) {
  return divUp(a, b) * b;
}

template <typename T>
bool isEvenDivisor(T a, T b) {
  return (a % b == 0) && ((a / b) >= 1);
}

template <class T>
constexpr T pow(T n, T power) {
  return (power > 0 ? n * pow(n, power - 1) : 1);
}

template <class T>
constexpr T pow2(T n) {
  return pow(2, (T)n);
}

static_assert(pow2(8) == 256, "pow2");

template <typename T>
constexpr int log2(T n, int p = 0) {
  return (n <= 1) ? p : log2(n / 2, p + 1);
}

static_assert(log2(2) == 1, "log2");
static_assert(log2(3) == 1, "log2");
static_assert(log2(4) == 2, "log2");

template <typename T>
constexpr bool isPowerOf2(T v) {
  return (v && !(v & (v - 1)));
}

static_assert(isPowerOf2(2048), "isPowerOf2");
static_assert(!isPowerOf2(3333), "isPowerOf2");

template <typename T>
constexpr T nextHighestPowerOf2(T v) {
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
constexpr T nextLowestPowerOf2(T v) {
  return (isPowerOf2(v) ? v / (T)2 : ((T)1 << (log2(v))));
}

static_assert(nextLowestPowerOf2(1) == 0, "nextLowestPowerOf2");
static_assert(nextLowestPowerOf2(2) == 1, "nextLowestPowerOf2");
static_assert(nextLowestPowerOf2(3) == 2, "nextLowestPowerOf2");
static_assert(nextLowestPowerOf2(4) == 2, "nextLowestPowerOf2");

static_assert(nextLowestPowerOf2(15) == 8, "nextLowestPowerOf2");
static_assert(nextLowestPowerOf2(16) == 8, "nextLowestPowerOf2");
static_assert(nextLowestPowerOf2(17) == 16, "nextLowestPowerOf2");

bool isPointerAligned(const void* p, int align) {
  return reinterpret_cast<uintptr_t>(p) % align == 0;
}

// Returns the increment needed to aligned the pointer to the next highest
// aligned address
template <int Align>
uint32_t getAlignmentRoundUp(const void* p) {
  static_assert(isPowerOf2(Align), "");
  uint32_t diff = uint32_t(uintptr_t(p) & uintptr_t(Align - 1));
  return diff == 0 ? 0 : uint32_t(Align) - diff;
}

template <typename T>
struct NoTransform {
    T operator()(const T& v) const {
        return v;
    }
};

struct ANSDecodedTx16 {
    ANSDecodedT x[16];
};

struct ANSDecodedTx8 {
    ANSDecodedT x[8];
};

struct ANSDecodedTx4 {
    ANSDecodedT x[4];
};

constexpr uint32_t kNumSymbols = 1 << (sizeof(ANSDecodedT) * 8);
static_assert(kNumSymbols > 1, "");

constexpr uint32_t kMaxBEPSThreads = 512;
// Default block size for compression (in bytes)
constexpr uint32_t kDefaultBlockSize = 4096;

constexpr int kANSDefaultProbBits = 10;

constexpr int kANSRequiredAlignment = 4;

// limit state to 2^31 - 1, so as to prevent addition overflow in the integer
// division via mul and shift by constants
constexpr int kANSStateBits = (sizeof(ANSStateT) * 8) - 1;
constexpr int kANSEncodedBits = sizeof(ANSEncodedT) * 8;
//constexpr ANSStateT kANSEncodedMask = (ANSStateT(1) << kANSEncodedBits) - 1;
constexpr ANSStateT kANSStartState = ANSStateT(1) << (kANSStateBits - kANSEncodedBits);
constexpr ANSStateT kANSMinState = ANSStateT(1) << (kANSStateBits - kANSEncodedBits);

constexpr ANSStateT kANSEncodedMask =
    (ANSStateT(1) << kANSEncodedBits) - ANSStateT(1);

// Magic number to verify archive integrity
constexpr uint32_t kANSMagic = 0xd00d;

// Current DietGPU version number
constexpr uint32_t kANSVersion = 0x0001;

// Each block of compressed data (either coalesced or uncoalesced) is aligned to
// this number of bytes and has a valid (if not all used) segment with this
// multiple of bytes
constexpr uint32_t kBlockAlignment = 16;

struct ANSCoalescedHeader {
    static uint32_t getCompressedOverhead(uint32_t numBlocks) {
        int kAlignment = kBlockAlignment / sizeof(uint2);
        if (kAlignment == 0) kAlignment = 1;

        return sizeof(ANSCoalescedHeader) +
               sizeof(uint16_t) * kNumSymbols +
               sizeof(ANSWarpState) * numBlocks +
               sizeof(uint2) * roundUp(numBlocks, kAlignment);
    }
    
    uint32_t getTotalCompressedSize() {
        return getCompressedOverhead() +
               getTotalCompressedWords() * sizeof(ANSEncodedT);
    }

    uint32_t getCompressedOverhead() {
        return getCompressedOverhead(getNumBlocks());
    }

    float getCompressionRatio() {
        return static_cast<float>(getTotalCompressedSize()) /
               static_cast<float>(getTotalUncompressedWords() * sizeof(ANSDecodedT));
    }

    uint32_t getNumBlocks() { return numBlocks; }
    void setNumBlocks(uint32_t nb) { numBlocks = nb; }

    void setMagicAndVersion() {
        magicAndVersion = (kANSMagic << 16) | kANSVersion;
    }

    void checkMagicAndVersion() {
        assert((magicAndVersion >> 16) == kANSMagic);
        assert((magicAndVersion & 0xffffU) == kANSVersion);
    }

    uint32_t getTotalUncompressedWords() { return totalUncompressedWords; }
    void setTotalUncompressedWords(uint32_t words) { totalUncompressedWords = words; }

    uint32_t getTotalCompressedWords() { return totalCompressedWords; }
    void setTotalCompressedWords(uint32_t words) { totalCompressedWords = words; }

    uint32_t getProbBits() { return options & 0xf; }
    void setProbBits(uint32_t bits) {
        assert(bits <= 0xf);
        options = (options & 0xfffffff0U) | bits;
    }

    bool getUseChecksum() { return options & 0x10; }
    void setUseChecksum(bool uc) {
        options = (options & 0xffffffef) | (static_cast<uint32_t>(uc) << 4);
    }

    uint32_t getChecksum() { return checksum; }
    void setChecksum(uint32_t c) { checksum = c; }

    uint16_t* getSymbolProbs() { return reinterpret_cast<uint16_t*>(this + 1); }

    ANSWarpState* getWarpStates() { return reinterpret_cast<ANSWarpState*>(getSymbolProbs() + kNumSymbols); }
 
    uint2* getBlockWords(uint32_t numBlocks) {
        return reinterpret_cast<uint2*>(getWarpStates() + numBlocks);
    }

    ANSEncodedT* getBlockDataStart(uint32_t numBlocks) {
        constexpr int kAlignment = kBlockAlignment / sizeof(uint2) == 0
        ? 1 : kBlockAlignment / sizeof(uint2);
        return reinterpret_cast<ANSEncodedT*>(getBlockWords(numBlocks) + roundUp(numBlocks, kAlignment));
    }

    uint32_t magicAndVersion;
    uint32_t numBlocks;
    uint32_t totalUncompressedWords;
    uint32_t totalCompressedWords;
    uint32_t options;
    uint32_t checksum;
    uint32_t unused0;
    uint32_t unused1;
};

static_assert(sizeof(ANSCoalescedHeader) == 32, "");

static_assert( (sizeof(ANSCoalescedHeader) % sizeof(uint4)) == 0, "" );

struct BatchWriter {
  BatchWriter(void* out)
      : out_((uint8_t*)out), outBlock_(nullptr) {}

  void setBlock(uint32_t block) {
    outBlock_ = out_ + block * kDefaultBlockSize;
  }

  void write(uint32_t offset, uint8_t sym) {
    outBlock_[offset] = sym;
  }

  // template <typename Vec>
  // void writeVec(uint32_t offset, Vec symV) {
  //   ((Vec*)outBlock_)[offset] = symV;
  // }

  // void preload(uint32_t offset) {}

  uint8_t* out_;
  uint8_t* outBlock_;
};

// maximum raw compressed data block size in bytes
//constexpr  
uint32_t
getRawCompBlockMaxSize(uint32_t uncompressedBlockBytes) {
  // (an estimate from zstd)
  return roundUp(
      uncompressedBlockBytes + (uncompressedBlockBytes / 4), kBlockAlignment);
}

uint32_t getMaxBlockSizeCoalesced(uint32_t uncompressedBlockBytes) {
  return getRawCompBlockMaxSize(uncompressedBlockBytes);
}

uint32_t getMaxCompressedSize(uint32_t uncompressedBytes) {
  uint32_t blocks = divUp(uncompressedBytes, kDefaultBlockSize);

  size_t rawSize = ANSCoalescedHeader::getCompressedOverhead(kDefaultBlockSize);
  rawSize += (size_t)getMaxBlockSizeCoalesced(kDefaultBlockSize) * blocks;

  // When used in batches, we must align everything to 16 byte boundaries (due
  // to uint4 read/writes)
  rawSize = roundUp(rawSize, sizeof(uint4));

  return rawSize;
}

uint32_t getMaxBlockSizeUnCoalesced(uint32_t uncompressedBlockBytes) {
  // uncoalesced data has a warp state header
  return sizeof(ANSWarpState) + getRawCompBlockMaxSize(uncompressedBlockBytes);
}

} // namespace dietgpu