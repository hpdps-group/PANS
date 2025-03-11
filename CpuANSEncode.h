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
#include "CpuANSUtils.h"

namespace cpu_ans {

void ansHistogram(
    const ANSDecodedT* in,
    uint32_t size,
    uint32_t* out) {

    for (int i = 0; i < kNumSymbols; ++i)
        out[i] = 0;

    uint32_t roundUp4 = std::min(size, getAlignmentRoundUp(sizeof(uint4), in));
    auto remaining = size - roundUp4;
    auto numU4 = divDown(remaining, sizeof(uint4));
    auto inAligned = in + roundUp4;
    auto inAligned4 = (const uint4*)inAligned;

    for(int i = 0; i < roundUp4; i ++){
      out[in[i]]++;
    }

    for(int i = 0; i < numU4; i ++){
      uint4 v = inAligned4[i];
      for(int j = 0; j < 4; j++){
        out[(v.x >> (8 * j)) & 0xFF]++;
        out[(v.y >> (8 * j)) & 0xFF]++;
        out[(v.z >> (8 * j)) & 0xFF]++;
        out[(v.w >> (8 * j)) & 0xFF]++;
      }
    }

    for(int i = numU4 * sizeof(uint4); i < remaining; i ++) {
      out[inAligned[i]]++;
    }
}

// 手动实现计算前导零数量的函数
uint32_t clz(uint32_t x) {
    if (x == 0) return 32; // 如果 x 为 0，返回 32

    uint32_t n = 0;
    while ((x & (1U << 31)) == 0) {
        x <<= 1;
        ++n;
    }
    return n;
}

// 概率归一化核心
void ansCalcWeights(
    int probBits,
    uint32_t totalNum,
    const uint32_t* counts,
    uint4* table) {

    if (totalNum == 0) return;

    const uint32_t kProbWeight = 1 << probBits;
    std::vector<uint32_t> qProb(kNumSymbols);
    std::vector<uint32_t> sortedPairs(kNumSymbols);

    // 初始化量化概率
    for (int i = 0; i < kNumSymbols; ++i) {
        qProb[i] = static_cast<uint32_t>(kProbWeight * (counts[i] / static_cast<float>(totalNum)));
        qProb[i] = (counts[i] > 0 && qProb[i] == 0) ? 1U : qProb[i];
        sortedPairs[i] = (qProb[i] << 16) | i;
    }

    // 基数排序（降序）
    std::sort(sortedPairs.begin(), sortedPairs.end(), [](uint32_t a, uint32_t b) {
        return a > b;
    });
    
    uint32_t tidSymbol[kNumSymbols];
    for (int i = 0; i < kNumSymbols; ++i) {
        tidSymbol[i] = sortedPairs[i] & 0xFFFFU;
        qProb[i] = sortedPairs[i] >> 16;
    }

    // 调整总概率和
    int currentSum = std::accumulate(qProb.begin(), qProb.end(), 0);
    int diff = static_cast<int>(kProbWeight) - currentSum;

    // 处理概率不足的情况
    if (diff > 0) {
      int iterToApply = std::min(diff, static_cast<int>(kNumSymbols));
      for(int i = diff; i > 0; i -= iterToApply){
        for(int j = 0; j < kNumSymbols; ++j){
          int cursym = tidSymbol[j];
          if(cursym < iterToApply){
            qProb[j] += 1;
          }          
        }
      }
    } 
    // 处理概率过载的情况（关键修正部分）
    else if (diff < 0) {
      diff = -diff;
      while(diff > 0){
        int qNumGt1s = 0;
        for(int j = 0; j < kNumSymbols; ++j){
          qNumGt1s += (int)(qProb[j] > 1);
        }
        int iterToApply = diff < qNumGt1s ? diff : qNumGt1s;
        int startIndex = qNumGt1s - iterToApply;
        for(int j = 0; j < kNumSymbols; ++j){
          if(j >= startIndex && j < qNumGt1s){
            qProb[j] -= 1;
          }
        }
        diff -= iterToApply;
      }  
    }

    uint32_t smemPdf[kNumSymbols];

    for(int i = 0; i < kNumSymbols; ++i){
      smemPdf[tidSymbol[i]] = qProb[i];
    }

    uint32_t symPdf[kNumSymbols];
    // 计算CDF
    for (int i = 0; i < kNumSymbols; ++i) {
        symPdf[i] = smemPdf[i];
    }

    // 独占前缀和
    std::vector<uint32_t> cdf(kNumSymbols, 0);
    for (int i = 1; i < kNumSymbols; ++i) {
        cdf[i] = cdf[i-1] + symPdf[i-1];
    }

    // 计算magic和shift
    for (int i = 0; i < kNumSymbols; ++i) {
        uint32_t p = symPdf[i];
        uint32_t shift = 32 - clz(p - 1);
        uint64_t magic = ((1ULL << 32) * ((1ULL << shift) - p)) / p + 1;
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
    // printf("inSize = %d, numBlocks = %d, maxNumCompressedBlocks = %d\n", inSize, numBlocks, maxNumCompressedBlocks);
// #pragma omp parallel for num_threads(8)
    #pragma omp parallel for num_threads(32) 
    for(int l = 0; l < maxNumCompressedBlocks; l ++){
      // if(l == 0 ) printf("l = %d\n", l);
      uint32_t start = l * BlockSize;

      // if(start >= inSize){
      //   return;
      // }

    auto blockSize =  std::min(start + BlockSize, (uint32_t)inSize) - start;
    // printf("l = %d, start = %d, blockSize = %d\n", l, start, blockSize);
    
    // if (l >= numBlocks)
    //   return;

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
    // if(l == 0)
    // printf("startstate = %d\n", state[0]);

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
          // auto pre_state = state[k];
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
          // calculating ((state / pdf) << ProbBits) + (state % pdf) + cdf
          constexpr uint32_t kProbBitsMul = 1 << ProbBits;
          state[k] = div * kProbBitsMul + mod + cdf;
          // if(l == 12265 && i == 15 && j == 7)
          // printf("pre_state[%d] = %d, state[%d] = %d, sym = %d, pdf = %d, cdf = %d, div_m1 = %d, div_shift = %d, offset = %d\n", k, pre_state, k, state[k], inBlock[inOffset[k] + j * kWarpSize], pdf, cdf, div_m1, div_shift, outOffset);
        }
    
        // how many values we actually write to the compressed output
        outOffset += count;
        // if(l == 12265 && i == 15 && j == 6){
        //   printf("outOffset = %d\n", outOffset);
        // }
        // if(l==0 && i == 0)
        // {
        //   printf("inOffset = %d, outOffset = %d\n",i * kWarpSize * kUnroll , outOffset);
        // }
        // outOffset +=
        //     encodeOne<ProbBits>(true, state, inBlock[inOffset + j * kWarpSize], outOffset, outWords, smemLookup);
      }
      for(int k = 0; k < kWarpSize; ++k){
        inOffset[k] += kWarpSize * kUnroll;
      }
    }
    // printf("limit = %d, blockSize = %d, cyclenum0 = %d, outOffset = %d\n", limit, blockSize, cyclenum0, outOffset);
  if (limit != blockSize) {
    uint32_t limit1 = roundDown(blockSize, kWarpSize);
    
    int cyclenum1 = (limit1 - limit) / kWarpSize;
    // printf("limit = %d, limit1 = %d, blockSize = %d, cyclenum1 = %d, outOffset = %d\n", limit, limit1, blockSize, cyclenum1, outOffset);
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
        // calculating ((state / pdf) << ProbBits) + (state % pdf) + cdf
        constexpr uint32_t kProbBitsMul = 1 << ProbBits;
        state[k] = div * kProbBitsMul + mod + cdf;
      }
      // how many values we actually write to the compressed output
      outOffset += count;
      for(int k = 0; k < kWarpSize; ++k){
        inOffset[k] += kWarpSize;
      }
    }
    if (limit1 != blockSize) {
      int count = 0;
      for(int k = 0; k < kWarpSize; ++k){
        // printf("k = %d, inOffset = %d, blockSize = %d\n", k, inOffset[k], blockSize);
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
        // calculating ((state / pdf) << ProbBits) + (state % pdf) + cdf
        constexpr uint32_t kProbBitsMul = 1 << ProbBits;
        state[k] = valid ? div * kProbBitsMul + mod + cdf : state[k];
      }
      // how many values we actually write to the compressed output
      outOffset += count;
    }
  }
  // Write final state at the beginning (aligned addresses)
  for(int i = 0; i < kWarpSize; ++i){
    outBlock->warpState[i] = state[i];
  }

  compressedWords_dev[l] = outOffset;
  // printf("l = %d, compressedWords_dev[l] = %d\n", l, compressedWords_dev[l]);
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

  auto numBlocks = divUp(uncompressedWords, kDefaultBlockSize);

  for(int i = 0; i < maxNumCompressedBlocks; i ++){
    ANSCoalescedHeader* headerOut = (ANSCoalescedHeader*)out;
    if(i == 0){
    uint32_t totalCompressedWords = 0;
    if(numBlocks > 0){
      totalCompressedWords =
          compressedWordsPrefix_host[numBlocks - 1] +
          roundUp(
              compressedWords_host[numBlocks - 1],
              kBlockAlignment / sizeof(ANSEncodedT));
    }
    
    ANSCoalescedHeader header;
    header.setMagicAndVersion();
    header.setNumBlocks(numBlocks);
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
    }         
    auto uncoalescedBlock = compressedBlocks_host + i * uncoalescedBlockStride;
    for(int j = 0; j < kWarpSize; ++j){
      auto warpStateOut = (ANSWarpState*)uncoalescedBlock;
      headerOut->getWarpStates()[i].warpState[j] = (warpStateOut->warpState[j]);
    }
    auto blockWordsOut = headerOut->getBlockWords(numBlocks);

    for(int j = 0; j < numBlocks; ++j){
      uint32_t lastBlockWords = uncompressedWords % kDefaultBlockSize;
      lastBlockWords = lastBlockWords == 0 ? kDefaultBlockSize : lastBlockWords;

      uint32_t blockWords =
          (j == numBlocks - 1) ? lastBlockWords : kDefaultBlockSize;

      blockWordsOut[j] = uint2{
          (blockWords << 16) | compressedWords_host[j], compressedWordsPrefix_host[j]};
    }

    uint32_t numWords = compressedWords_host[i];
    using LoadT = uint4;

    uint32_t limitEnd = divUp(numWords, kBlockAlignment / sizeof(ANSEncodedT));

    auto inT = (const LoadT*)(uncoalescedBlock + sizeof(ANSWarpState));
    auto outT =
        (LoadT*)(headerOut->getBlockDataStart(numBlocks) + compressedWordsPrefix_host[i]);
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
      (maxUncompressedWords + kDefaultBlockSize - 1) / kDefaultBlockSize;//一个batch的数据以kDefaultBlockSize作为基准划分数据，形成多个数据块
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
    
  // for(int i = 0; i < kNumSymbols; i ++){
  //   printf("table[%d] = {%d, %d, %d, %d}\n", i, table[i].x, table[i].y, table[i].z, table[i].w);
  // }
  
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
    // auto sizeRequired = divUp(maxNumCompressedBlocks, kMaxBEPSThreads);
    // uint8_t* tempPrefixSum_host = (uint8_t*)malloc(sizeof(uint8_t) * sizeRequired);
    // batchExclusivePrefixSum<uint32_t, Align<ANSEncodedT, kBlockAlignment>>(
    //     compressedWords_host,
    //     compressedWordsPrefix_host,
    //     tempPrefixSum_host,
    //     maxNumCompressedBlocks,
    //     Align<ANSEncodedT, kBlockAlignment>());
    compressedWordsPrefix_host[0] = 0;
    for(int i = 1; i < maxNumCompressedBlocks; i ++){
      compressedWordsPrefix_host[i] = compressedWordsPrefix_host[i - 1] + compressedWords_host[i - 1];
    }
    // printf("compressedWordsPrefix_host[12288] = %d\n", compressedWordsPrefix_host[12288]);
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