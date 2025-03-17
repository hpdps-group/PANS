/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef CPU_ANS_INCLUDE_ANS_CPUANSENCODE_H
#define CPU_ANS_INCLUDE_ANS_CPUANSENCODE_H

#pragma once

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>
#include <numeric> 
#include <immintrin.h>
#include <thread>
#include <parallel/algorithm>
#include <avx512fintrin.h>
// #include <parallel/sort>
#include "CpuANSUtils.h"

namespace cpu_ans {
#define ALIGN alignas(64)

void processBlock(const uint8_t* in, uint32_t size, uint32_t* localHist) {
    constexpr uint32_t kAlign = 32;

    uint32_t roundUp = std::min(size, static_cast<uint32_t>(getAlignmentRoundUp(kAlign, in)));
    for (uint32_t i = 0; i < roundUp; ++i) {
        localHist[in[i]]++;
    }

    const uint8_t* alignedIn = in + roundUp;
    uint32_t remaining = size - roundUp;
    uint32_t numChunks = remaining / kAlign;

    const __m256i* avxIn = reinterpret_cast<const __m256i*>(alignedIn);
    for (uint32_t i = 0; i < numChunks; ++i) {
        __m256i vec = _mm256_load_si256(avxIn + i);
        alignas(kAlign) uint8_t buffer[kAlign];
        _mm256_store_si256(reinterpret_cast<__m256i*>(buffer), vec);
        #pragma unroll(2)
        for (uint32_t j = 0; j < kAlign - 7; j += 8) {
            localHist[buffer[j]]++;
            localHist[buffer[j+1]]++;
            localHist[buffer[j+2]]++;
            localHist[buffer[j+3]]++;
            localHist[buffer[j+4]]++;
            localHist[buffer[j+5]]++;
            localHist[buffer[j+6]]++;
            localHist[buffer[j+7]]++;
        }
    }
    const uint8_t* tail = alignedIn + numChunks * kAlign;
    for (uint32_t i = 0; i < remaining % kAlign; ++i) {
        localHist[tail[i]]++;
    }
}

void ansHistogram(
    const uint8_t* in,
    uint32_t size,
    uint32_t* out,
    bool multithread = true) {

    memset(out, 0, kNumSymbols * sizeof(uint32_t));

    if (size < 100000 || !multithread) {
        uint32_t localHist[kNumSymbols] = {0};
        processBlock(in, size, localHist);
        for (int i = 0; i < kNumSymbols; ++i) {
            out[i] += localHist[i];
        }
        return;
    }

    const unsigned numThreads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;
    std::vector<uint32_t> histograms(numThreads * kNumSymbols, 0);

    const uint32_t blockSize = (size + numThreads - 1) / numThreads;

    for (unsigned t = 0; t < numThreads; ++t) {
        threads.emplace_back([t, &histograms, in, size, blockSize]() {
            uint32_t* localHist = &histograms[t * kNumSymbols];
            const uint32_t start = std::min(t * blockSize, size);
            const uint32_t end = std::min((t + 1) * blockSize, size);
            processBlock(in + start, end - start, localHist);
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }
    
    for (unsigned t = 0; t < numThreads; ++t) {
        const uint32_t* localHist = &histograms[t * kNumSymbols];
        #pragma unroll(2)
        for (int i = 0; i < kNumSymbols - 1; i += 2) {
            out[i] += localHist[i];
            out[i+1] += localHist[i+1];
        }
    }
}   

void ansCalcWeights(
    int probBits,
    uint32_t totalNum,
    const uint32_t* counts,
    uint4* table) {
    if (totalNum == 0) return;
    const uint32_t kProbWeight = 1 << probBits;
    std::vector<uint32_t> qProb(kNumSymbols);
    std::vector<uint32_t> sortedPairs(kNumSymbols);
    for (int i = 0; i < kNumSymbols; ++i) {
        qProb[i] = static_cast<uint32_t>(kProbWeight * (counts[i] / static_cast<float>(totalNum)));
        qProb[i] = (counts[i] > 0 && qProb[i] == 0) ? 1U : qProb[i];
        sortedPairs[i] = (qProb[i] << 16) | i;
    }
    #pragma omp single
    {
        __gnu_parallel::sort(
            sortedPairs.begin(), 
            sortedPairs.end(),
            [](uint32_t a, uint32_t b) { return a > b; },
            __gnu_parallel::balanced_quicksort_tag()
        );
    }
    uint32_t tidSymbol[kNumSymbols];
    for (int i = 0; i < kNumSymbols; ++i) {
        tidSymbol[i] = sortedPairs[i] & 0xFFFFU;
        qProb[i] = sortedPairs[i] >> 16;
    }
    int currentSum = 0;
    #pragma omp parallel num_threads(32)
    {
        int localSum = 0;
        #pragma omp for schedule(static)
        for (int i = 0; i < kNumSymbols; ++i) {
            localSum += qProb[i];
        }
        #pragma omp atomic
        currentSum += localSum;
    }
    int diff = static_cast<int>(kProbWeight) - currentSum;
    if (diff > 0) {
      int iterToApply = std::min(diff, static_cast<int>(kNumSymbols));
      for(int i = diff; i > 0; i -= iterToApply){
        #pragma omp parallel for num_threads(32) schedule(static)
        for(int j = 0; j < kNumSymbols; ++j){
            qProb[j] += (tidSymbol[j] < iterToApply);       
        }
      }
    }
    else if (diff < 0) {
      diff = -diff;
      while(diff > 0){
        int qNumGt1s = 0;
        for(int j = 0; j < kNumSymbols; ++j){
          qNumGt1s += (int)(qProb[j] > 1);
        }
        int iterToApply = diff < qNumGt1s ? diff : qNumGt1s;
        int startIndex = qNumGt1s - iterToApply;
        #pragma omp parallel for num_threads(32) schedule(static)
        for(int j = 0; j < kNumSymbols; ++j){
          if(j >= startIndex && j < qNumGt1s){
            qProb[j] -= 1;
          }
        }
        diff -= iterToApply;
      }  
    }
    uint32_t symPdf[kNumSymbols];
    #pragma omp for simd schedule(static)
    for(int i = 0; i < kNumSymbols; i ++){
      symPdf[tidSymbol[i]] = qProb[i];
    }
    std::vector<uint32_t> cdf(kNumSymbols, 0);
    uint32_t pp = symPdf[0];
    uint32_t shift0 = 32 - __builtin_clz(pp - 1);
    uint64_t magic0 = ((1ULL << 32) * ((1ULL << shift0) - pp)) / pp + 1;
    table[0] = {pp, 0, static_cast<uint32_t>(magic0), shift0};
    for (int i = 1; i < kNumSymbols; ++i) {
        uint32_t p = symPdf[i];
        uint32_t shift = 32 - __builtin_clz(p - 1);
        uint64_t magic = ((1ULL << 32) * ((1ULL << shift) - p)) / p + 1;
        cdf[i] = cdf[i-1] + symPdf[i-1];
        table[i] = {p, cdf[i], static_cast<uint32_t>(magic), shift};
    }
}

uint32_t umulhi(uint32_t a, uint32_t b) {
    uint64_t res = (uint64_t)a * (uint64_t)b;
    uint32_t hi = (uint32_t)(res >> 32);
    return hi;
}

template <int ProbBits, int BlockSize>
void ansEncodeBatch(
    uint8_t* in,
    int inSize,
    uint32_t maxNumCompressedBlocks,
    uint32_t uncoalescedBlockStride,
    uint8_t* compressedBlocks_dev,
    uint32_t* compressedWords_dev,
    const uint4* table) {
    uint32_t numBlocks = (inSize + BlockSize - 1) / BlockSize;
    #pragma omp parallel for num_threads(32) 
    for(int l = 0; l < maxNumCompressedBlocks; l ++){
      uint32_t start = l * BlockSize;
    auto blockSize =  std::min(start + BlockSize, (uint32_t)inSize) - start;
    auto inBlock = in + start;
    auto outBlock = (ANSWarpState*)(compressedBlocks_dev
        + l * uncoalescedBlockStride);
    assert(isPointerAligned(inBlock, kANSRequiredAlignment));
    ANSEncodedT* outWords = (ANSEncodedT*)(outBlock + 1);
    uint32_t inOffset[kWarpSize];
    for(int i = 0; i < kWarpSize; ++i){
      inOffset[i] = i;
    }
    uint32_t state[kWarpSize];
    for(int i = 0; i < kWarpSize; ++i){
      state[i] = kANSStartState;
    }
    uint32_t outOffset = 0;
    constexpr int kUnroll = 8;
    uint32_t limit = roundDown(blockSize, kWarpSize * kUnroll);
    int cyclenum0 = limit / (kWarpSize * kUnroll);
    for (int i = 0; i < cyclenum0; ++i) {
      for (int j = 0; j < kUnroll; ++j) {
        int count = 0;
        for(int k = 0; k < kWarpSize; ++k){
          auto lookup = table[inBlock[inOffset[k] + j * kWarpSize]];
          uint32_t pdf = lookup.x;
          uint32_t cdf = lookup.y;
          uint32_t div_m1 = lookup.z;
          uint32_t div_shift = lookup.w;
          constexpr ANSStateT kStateCheckMul = 1 << (kANSStateBits - ProbBits);
          ANSStateT maxStateCheck = pdf * kStateCheckMul;
          bool write = (state[k] >= maxStateCheck);
          if (write) {
            outWords[outOffset + count] = state[k] & kANSEncodedMask;
            state[k] >>= kANSEncodedBits;
            count ++;
          }
          uint32_t t = umulhi(state[k], div_m1);
          uint32_t div = (t + state[k]) >> div_shift;
          auto mod = state[k] - (div * pdf);
          constexpr uint32_t kProbBitsMul = 1 << ProbBits;
          state[k] = div * kProbBitsMul + mod + cdf;
        }
        outOffset += count;
      }
      for(int k = 0; k < kWarpSize; ++k){
        inOffset[k] += kWarpSize * kUnroll;
      }
    }
  if (limit != blockSize) {
    uint32_t limit1 = roundDown(blockSize, kWarpSize);
    
    int cyclenum1 = (limit1 - limit) / kWarpSize;
    for(int i = 0; i < cyclenum1; ++i){
      int count = 0;
      for(int k = 0; k < kWarpSize; ++k){
        auto lookup = table[inBlock[inOffset[k]]];
        uint32_t pdf = lookup.x;
        uint32_t cdf = lookup.y;
        uint32_t div_m1 = lookup.z;
        uint32_t div_shift = lookup.w;
        constexpr ANSStateT kStateCheckMul = 1 << (kANSStateBits - ProbBits);
        ANSStateT maxStateCheck = pdf * kStateCheckMul;
        bool write = (state[k] >= maxStateCheck);
        if (write) {
          outWords[outOffset + count] = state[k] & kANSEncodedMask;
          state[k] >>= kANSEncodedBits;
          count ++;
        }
        uint32_t t = umulhi(state[k], div_m1);
        uint32_t div = (t + state[k]) >> div_shift;
        auto mod = state[k] - (div * pdf);
        constexpr uint32_t kProbBitsMul = 1 << ProbBits;
        state[k] = div * kProbBitsMul + mod + cdf;
      }
      outOffset += count;
      for(int k = 0; k < kWarpSize; ++k){
        inOffset[k] += kWarpSize;
      }
    }
    if (limit1 != blockSize) {
      int count = 0;
      for(int k = 0; k < kWarpSize; ++k){
        bool valid = inOffset[k] < blockSize;
        ANSDecodedT sym = valid ? inBlock[inOffset[k]] : ANSDecodedT(0);
        auto lookup = table[sym];
        uint32_t pdf = lookup.x;
        uint32_t cdf = lookup.y;
        uint32_t div_m1 = lookup.z;
        uint32_t div_shift = lookup.w;
        constexpr ANSStateT kStateCheckMul = 1 << (kANSStateBits - ProbBits);

        ANSStateT maxStateCheck = pdf * kStateCheckMul;
        bool write = valid && (state[k] >= maxStateCheck);
        if (write) {
          outWords[outOffset + count] = state[k] & kANSEncodedMask;
          state[k] >>= kANSEncodedBits;
          count ++;
        }
        uint32_t t = umulhi(state[k], div_m1);
        uint32_t div = (t + state[k]) >> div_shift;
        auto mod = state[k] - (div * pdf);
        constexpr uint32_t kProbBitsMul = 1 << ProbBits;
        state[k] = valid ? div * kProbBitsMul + mod + cdf : state[k];
      }
      outOffset += count;
    }
  }
  for(int i = 0; i < kWarpSize; ++i){
    outBlock->warpState[i] = state[i];
  }
  compressedWords_dev[l] = outOffset;
  }
}

template <typename A, int B>
struct Align {
  typedef uint32_t argument_type;
  typedef uint32_t result_type;
  template <typename T> uint32_t operator()(T x) const {
    constexpr int kDiv = B / sizeof(A);
    constexpr int kSize = kDiv < 1 ? 1 : kDiv;
    return roundUp(x, T(kSize));
  }
};

void ansEncodeCoalesceBatch(
    const uint8_t* __restrict__ compressedBlocks_host,
    int uncompressedWords,
    uint32_t maxNumCompressedBlocks,
    uint32_t uncoalescedBlockStride,
    const uint32_t* __restrict__ compressedWords_host,
    const uint32_t* __restrict__ compressedWordsPrefix_host,
    const uint4* __restrict__ table,
    uint32_t config_probBits,
    uint8_t* out,
    uint32_t* outSize) {

  ANSCoalescedHeader* headerOut = (ANSCoalescedHeader*)out;
  uint32_t totalCompressedWords = 0;
  if(maxNumCompressedBlocks > 0){
    totalCompressedWords =
        compressedWordsPrefix_host[maxNumCompressedBlocks - 1] +
            roundUp(
            compressedWords_host[maxNumCompressedBlocks - 1],
            kBlockAlignment / sizeof(ANSEncodedT));
  }
    
  ANSCoalescedHeader header;
  header.setMagicAndVersion();
  header.setNumBlocks(maxNumCompressedBlocks);
  header.setTotalUncompressedWords(uncompressedWords);
  header.setTotalCompressedWords(totalCompressedWords);
  header.setProbBits(config_probBits);

  if (outSize) {
    *outSize = header.getTotalCompressedSize();
  }
  *headerOut = header;

  auto probsOut = headerOut->getSymbolProbs();

  // Write out pdf
  for (int j = 0; j < kNumSymbols; j ++) {
    probsOut[j] = table[j].x;
  }
  auto blockWordsOut = headerOut->getBlockWords(maxNumCompressedBlocks);

  // #pragma omp parallel for proc_bind(spread) num_threads(32)
  for(int i = 0; i < maxNumCompressedBlocks; i ++){
    
    auto uncoalescedBlock = compressedBlocks_host + i * uncoalescedBlockStride;
    for(int j = 0; j < kWarpSize; ++j){
      auto warpStateOut = (ANSWarpState*)uncoalescedBlock;
      headerOut->getWarpStates()[i].warpState[j] = (warpStateOut->warpState[j]);
    }
    
    uint32_t lastBlockWords = uncompressedWords % kDefaultBlockSize;
    lastBlockWords = lastBlockWords == 0 ? kDefaultBlockSize : lastBlockWords;

    uint32_t blockWords =
        (i == maxNumCompressedBlocks - 1) ? lastBlockWords : kDefaultBlockSize;

    blockWordsOut[i] = uint2{
        (blockWords << 16) | compressedWords_host[i], compressedWordsPrefix_host[i]};

    uint32_t numWords = compressedWords_host[i];
    using LoadT = uint4;

    uint32_t limitEnd = divUp(numWords, kBlockAlignment / sizeof(ANSEncodedT));

    auto inT = (const LoadT*)(uncoalescedBlock + sizeof(ANSWarpState));
    auto outT =
        (LoadT*)(headerOut->getBlockDataStart(maxNumCompressedBlocks) + compressedWordsPrefix_host[i]);
    for(int j = 0; j < limitEnd; ++j){
      outT[j] = inT[j];
    }
  }
}


void ansEncode(
    int precision,
    uint8_t* in,
    uint32_t inSize,
    uint8_t* out,
    uint32_t* outSize) {
  uint32_t maxUncompressedWords = inSize / sizeof(ANSDecodedT);
  uint32_t maxNumCompressedBlocks =
      (maxUncompressedWords + kDefaultBlockSize - 1) / kDefaultBlockSize;
  uint4* table = (uint4*)malloc(sizeof(uint4) * kNumSymbols);
  uint32_t* tempHistogram = (uint32_t*)malloc(sizeof(uint32_t) * kNumSymbols);
  ansHistogram(
      in,
      inSize,
      tempHistogram);
  ansCalcWeights(
      precision,
      inSize,
      tempHistogram,
      table);
  uint32_t uncoalescedBlockStride = getMaxBlockSizeUnCoalesced(kDefaultBlockSize);
  uint8_t* compressedBlocks_host = (uint8_t*)malloc(sizeof(uint8_t) * maxNumCompressedBlocks * uncoalescedBlockStride);
  uint32_t* compressedWords_host = (uint32_t*)malloc(sizeof(uint32_t) * maxNumCompressedBlocks);
  uint32_t* compressedWordsPrefix_host = (uint32_t*)malloc(sizeof(uint32_t) * maxNumCompressedBlocks);

#define RUN_ENCODE(BITS)                                       \
  do {                                                         \
    ansEncodeBatch<BITS, kDefaultBlockSize> (                   \
            in,\
            inSize,                                        \
            maxNumCompressedBlocks,                            \
            uncoalescedBlockStride,                            \
            compressedBlocks_host,                       \
            compressedWords_host,                        \
            table);                                 \
  } while (false)

    switch (precision) {
      case 9:
        RUN_ENCODE(9);
        break;
      case 10:
        RUN_ENCODE(10);
        break;
      case 11:
        RUN_ENCODE(11);
        break;
      default:
        std::cout<< "unhandled pdf precision " << precision << std::endl;
    }
#undef RUN_ENCODE

  if (maxNumCompressedBlocks > 0) {
    compressedWordsPrefix_host[0] = 0;
    for(int i = 1; i < maxNumCompressedBlocks; i ++){
      compressedWordsPrefix_host[i] = compressedWordsPrefix_host[i - 1] + compressedWords_host[i - 1];
    }
  }
  ansEncodeCoalesceBatch(
          compressedBlocks_host,
          inSize,
          maxNumCompressedBlocks,
          uncoalescedBlockStride,
          compressedWords_host,
          compressedWordsPrefix_host,
          table,
          precision,
          out,
          outSize);
}
} // namespace 
#undef RUN_ENCODE_ALL
#endif