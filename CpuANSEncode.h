/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef CPU_ANS_INCLUDE_ANS_CPUANSENCODE_H
#define CPU_ANS_INCLUDE_ANS_CPUANSENCODE_H

#pragma once

#include "CpuANSUtils.h"

namespace cpu_ans {

constexpr uint32_t kAlign = 32;
uint32_t umulhi(uint32_t a, uint32_t b) {
    uint64_t product = (uint64_t)a * (uint64_t)b;
    return (uint32_t)(product >> 32);
}

template <size_t Align>
inline uint32_t getAlignmentRoundUp(const void* ptr) {
  const auto address = reinterpret_cast<size_t>(ptr);
  return (Align - (address % Align)) % Align;
}

void ansHistogram(
    const uint8_t* input,
    uint32_t size,
    uint32_t* histogram) 
{
  constexpr size_t kBatchSize = 4096;
  std::vector<std::atomic<uint32_t>> atomic_counts(kNumSymbols);
  auto worker = [&](size_t start_idx, size_t end_idx) {
    uint32_t local_counts[kNumSymbols] = {0};
    const uint32_t align_offset = getAlignmentRoundUp<sizeof(uint64_t)>(input);
    const size_t aligned_start = std::min(end_idx, start_idx + align_offset);
    for (size_t i = start_idx; i < aligned_start; ++i) {
      ++local_counts[input[i]];
    }
    const uint64_t* aligned_ptr = reinterpret_cast<const uint64_t*>(
        input + align_offset);
    const size_t vector_items = (end_idx - aligned_start) / sizeof(uint64_t);
    for (size_t i = 0; i < vector_items; ++i) {
      uint64_t packed = aligned_ptr[i];
      for (int shift = 0; shift < 64; shift += 8) {
        ++local_counts[(packed >> shift) & 0xFF];
      }
    }
    const size_t tail_start = aligned_start + vector_items * sizeof(uint64_t);
    for (size_t i = tail_start; i < end_idx; ++i) {
      ++local_counts[input[i]];
    }
    for (int k = 0; k < kNumSymbols; ++k) {
      if (local_counts[k] > 0) {
        atomic_counts[k].fetch_add(local_counts[k], std::memory_order_relaxed);
      }
    }
  };
  const size_t num_workers = std::thread::hardware_concurrency();
  std::vector<std::thread> workers;
  const size_t chunk_size = (size + num_workers - 1) / num_workers;
  for (size_t t = 0; t < num_workers; ++t) {
    const size_t start = t * chunk_size;
    const size_t end = std::min(start + chunk_size, static_cast<size_t>(size));
    workers.emplace_back(worker, start, end);
  }
  for (auto& w : workers) w.join();
  for (int k = 0; k < kNumSymbols; ++k) {
    histogram[k] = atomic_counts[k].load();
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
        uint64_t magic = 0.0;
        if(p != 0)
          magic = ((1ULL << 32) * ((1ULL << shift) - p)) / p + 1;
        cdf[i] = cdf[i-1] + symPdf[i-1];
        table[i] = {p, cdf[i], static_cast<uint32_t>(magic), shift};
    }
}

template <int ProbBits, int BlockSize>
void ansEncodeBatch(
    uint8_t* in,
    int inSize,
    uint32_t maxNumCompressedBlocks,
    uint32_t uncoalescedBlockStride,
    uint8_t* compressedBlocks_dev,
    uint32_t* compressedWords_dev,
    uint32_t* compressedWordsPrefix_host,
    const uint4* table) {
    constexpr ANSStateT kStateCheckMul = kANSStateBits - ProbBits;
    #pragma omp parallel for proc_bind(spread) num_threads(32) 
    for(int l = 0; l < maxNumCompressedBlocks; l ++){
    uint32_t start = l * BlockSize;
    auto blockSize =  std::min(start + BlockSize, (uint32_t)inSize) - start;
    auto inBlock = in + start;
    auto outBlock = (ANSWarpState*)(compressedBlocks_dev
        + l * uncoalescedBlockStride);
    ANSEncodedT* outWords = (ANSEncodedT*)(outBlock + 1);
    uint32_t inOffset[kWarpSize];
    uint32_t state[kWarpSize];
    #pragma unroll(4)
    for(int i = 0; i < kWarpSize; ++i){
      inOffset[i] = i;
      state[i] = kANSStartState;
    }
    uint32_t outOffset = 0;
    uint32_t limit = roundDown(blockSize, kWarpSize << 3);
    int cyclenum0 = limit / (kWarpSize << 3);
    for (int i = 0; i < cyclenum0; ++i) {
      for (int j = 0; j < 8; ++j) {
        for(int k = 0; k < kWarpSize; ++k){
          auto lookup = table[inBlock[inOffset[k] + j * kWarpSize]];
          uint32_t pdf = lookup.x;
          ANSStateT maxStateCheck = pdf << kStateCheckMul;
          uint32_t write = (state[k] >= maxStateCheck);
          if (write) {
            outWords[outOffset] = (state[k] & kANSEncodedMask);
            state[k] = (state[k] >> kANSEncodedBits);
            outOffset ++;
          }
          uint32_t div = (umulhi(state[k], lookup.z) + state[k]) >> lookup.w;
          state[k] = (div << ProbBits) + state[k] - (div * pdf) + lookup.y;
        }
      }
      for(int k = 0; k < kWarpSize; ++k){
        inOffset[k] += kWarpSize << 3;
      }
    }
  if (blockSize - limit) {
    uint32_t limit1 = roundDown(blockSize, kWarpSize);
    int cyclenum1 = (limit1 - limit) / kWarpSize;
    for(int i = 0; i < cyclenum1; ++i){
      for(int k = 0; k < kWarpSize; ++k){
          auto lookup = table[inBlock[inOffset[k]]];
          uint32_t pdf = lookup.x;
          ANSStateT maxStateCheck = pdf << kStateCheckMul;
          uint32_t write = (state[k] >= maxStateCheck);
          if (write) {
            outWords[outOffset] = (state[k] & kANSEncodedMask);
            state[k] = (state[k] >> kANSEncodedBits);
            outOffset ++;
          }
          uint32_t div = (umulhi(state[k], lookup.z) + state[k]) >> lookup.w;
          state[k] = (div << ProbBits) + state[k] - (div * pdf) + lookup.y;
          inOffset[k] += kWarpSize;
      }
    }
    if (blockSize - limit1) {
      for(int k = 0; k < kWarpSize && inOffset[k] < blockSize; ++k){
          auto lookup = table[inBlock[inOffset[k]]];
          uint32_t pdf = lookup.x;
          ANSStateT maxStateCheck = pdf << kStateCheckMul;
          uint32_t write = (state[k] >= maxStateCheck);
          if (write) {
            outWords[outOffset] = (state[k] & kANSEncodedMask);
            state[k] = (state[k] >> kANSEncodedBits);
            outOffset ++;
          }
          uint32_t div = (umulhi(state[k], lookup.z) + state[k]) >> lookup.w;
          state[k] = (div << ProbBits) + state[k] - (div * pdf) + lookup.y;
      }
    }
  }
  for(int i = 0; i < kWarpSize; ++i){
    outBlock->warpState[i] = state[i];
  }
  compressedWords_dev[l] = outOffset;
  }
}

void ansEncodeCoalesceBatch(
    const uint8_t* __restrict__ compressedBlocks_host,
    int uncompressedWords,
    uint32_t maxNumCompressedBlocks,
    uint32_t uncoalescedBlockStride,
    const uint32_t* __restrict__ compressedWords_host,
    uint32_t* __restrict__ compressedWordsPrefix_host,
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

  *outSize = header.getTotalCompressedSize();
  *headerOut = header;

  auto probsOut = headerOut->getSymbolProbs();
  for (int j = 0; j < kNumSymbols; j ++) {
    probsOut[j] = table[j].x;
  }
  auto blockWordsOut = headerOut->getBlockWords(maxNumCompressedBlocks);
  auto BlockDataStart = headerOut->getBlockDataStart(maxNumCompressedBlocks);
  compressedWordsPrefix_host[0] = 0;
  int i = 0;
  for(; i < maxNumCompressedBlocks - 1; i ++){
    
    auto uncoalescedBlock = compressedBlocks_host + i * uncoalescedBlockStride;
    for(int j = 0; j < kWarpSize; ++j){
      auto warpStateOut = (ANSWarpState*)uncoalescedBlock;
      headerOut->getWarpStates()[i].warpState[j] = (warpStateOut->warpState[j]);
    }

    blockWordsOut[i] = uint2{
        (kDefaultBlockSize << 16) | compressedWords_host[i], compressedWordsPrefix_host[i]};

    uint32_t numWords = compressedWords_host[i];

    uint32_t limitEnd = divUp(numWords, kBlockAlignment / sizeof(ANSEncodedT));

    auto inT = (const uint4*)(uncoalescedBlock + sizeof(ANSWarpState));
    auto outT = (uint4*)(BlockDataStart + compressedWordsPrefix_host[i]);

    memcpy(outT, inT, limitEnd << 4);
  }
  auto uncoalescedBlock = compressedBlocks_host + i * uncoalescedBlockStride;
  for(int j = 0; j < kWarpSize; ++j){
    auto warpStateOut = (ANSWarpState*)uncoalescedBlock;
    headerOut->getWarpStates()[i].warpState[j] = (warpStateOut->warpState[j]);
  }
    
  uint32_t lastBlockWords = uncompressedWords % kDefaultBlockSize;
  lastBlockWords = lastBlockWords == 0 ? kDefaultBlockSize : lastBlockWords;

  blockWordsOut[i] = uint2{
      (lastBlockWords << 16) | compressedWords_host[i], compressedWordsPrefix_host[i]};

  uint32_t numWords = compressedWords_host[i];

  uint32_t limitEnd = divUp(numWords, kBlockAlignment / sizeof(ANSEncodedT));

  auto inT = (const uint4*)(uncoalescedBlock + sizeof(ANSWarpState));
  auto outT = (uint4*)(BlockDataStart + compressedWordsPrefix_host[i]);

  memcpy(outT, inT, limitEnd << 4);
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
  uint8_t* compressedBlocks_host = (uint8_t*)std::aligned_alloc(kBlockAlignment, sizeof(uint8_t) * maxNumCompressedBlocks * uncoalescedBlockStride);
  uint32_t* compressedWords_host = (uint32_t*)std::aligned_alloc(kBlockAlignment, sizeof(uint32_t) * maxNumCompressedBlocks);
  uint32_t* compressedWordsPrefix_host = (uint32_t*)std::aligned_alloc(kBlockAlignment, sizeof(uint32_t) * maxNumCompressedBlocks);

#define RUN_ENCODE(BITS)                                       \
  do {                                                         \
    ansEncodeBatch<BITS, kDefaultBlockSize> (                   \
            in,\
            inSize,                                        \
            maxNumCompressedBlocks,                            \
            uncoalescedBlockStride,                            \
            compressedBlocks_host,                       \
            compressedWords_host,         \
            compressedWordsPrefix_host,                       \
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