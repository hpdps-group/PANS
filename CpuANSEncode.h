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

__attribute__((target("avx2")))
void processBlock(const __restrict uint8_t* in, uint32_t size, uint32_t* __restrict localHist) {
    uint32_t roundUp = std::min(size, static_cast<uint32_t>(getAlignmentRoundUp(kAlign, in)));
    for (uint32_t i = 0; i < roundUp; ++i) {
        ++localHist[in[i]];
    }

    const uint8_t* alignedIn = in + roundUp;
    uint32_t remaining = size - roundUp;
    uint32_t numChunks = remaining / kAlign;

    const __m256i* avxIn = reinterpret_cast<const __m256i*>(alignedIn);
    for (uint32_t i = 0; i < numChunks; ++i) {
        const __m256i vec = _mm256_load_si256(avxIn + i);
        const uint8_t* bytes = reinterpret_cast<const uint8_t*>(avxIn + i);
        
        _mm_prefetch(reinterpret_cast<const char*>(avxIn + i + 1), _MM_HINT_T0);
        
        uint32_t v0 = bytes[0], v1 = bytes[1], v2 = bytes[2], v3 = bytes[3];
        uint32_t v4 = bytes[4], v5 = bytes[5], v6 = bytes[6], v7 = bytes[7];
        ++localHist[v0]; ++localHist[v1]; ++localHist[v2]; ++localHist[v3];
        ++localHist[v4]; ++localHist[v5]; ++localHist[v6]; ++localHist[v7];

        uint32_t v8 = bytes[8], v9 = bytes[9], v10 = bytes[10], v11 = bytes[11];
        uint32_t v12 = bytes[12], v13 = bytes[13], v14 = bytes[14], v15 = bytes[15];
        ++localHist[v8]; ++localHist[v9]; ++localHist[v10]; ++localHist[v11];
        ++localHist[v12]; ++localHist[v13]; ++localHist[v14]; ++localHist[v15];

        uint32_t v16 = bytes[16], v17 = bytes[17], v18 = bytes[18], v19 = bytes[19];
        uint32_t v20 = bytes[20], v21 = bytes[21], v22 = bytes[22], v23 = bytes[23];
        ++localHist[v16]; ++localHist[v17]; ++localHist[v18]; ++localHist[v19];
        ++localHist[v20]; ++localHist[v21]; ++localHist[v22]; ++localHist[v23];

        uint32_t v24 = bytes[24], v25 = bytes[25], v26 = bytes[26], v27 = bytes[27];
        uint32_t v28 = bytes[28], v29 = bytes[29], v30 = bytes[30], v31 = bytes[31];
        ++localHist[v24]; ++localHist[v25]; ++localHist[v26]; ++localHist[v27];
        ++localHist[v28]; ++localHist[v29]; ++localHist[v30]; ++localHist[v31];
    }

    const uint8_t* tail = alignedIn + numChunks * kAlign;
    uint32_t remainingTail = remaining % kAlign;
    
    if (remainingTail >= 8) {
        const uint8_t* chunk = tail;
        ++localHist[chunk[0]]; ++localHist[chunk[1]]; 
        ++localHist[chunk[2]]; ++localHist[chunk[3]];
        ++localHist[chunk[4]]; ++localHist[chunk[5]];
        ++localHist[chunk[6]]; ++localHist[chunk[7]];
        tail += 8;
        remainingTail -= 8;
    }

    switch (remainingTail) {
        case 7: ++localHist[tail[6]];
        case 6: ++localHist[tail[5]];
        case 5: ++localHist[tail[4]];
        case 4: ++localHist[tail[3]];
        case 3: ++localHist[tail[2]];
        case 2: ++localHist[tail[1]];
        case 1: ++localHist[tail[0]];
        default: break;
    }
}

void ansHistogram(
    const uint8_t* __restrict in,
    uint32_t size,
    uint32_t* __restrict out,
    bool multithread = true) {
      std::memset(out, 0, kNumSymbols * sizeof(uint32_t));

    if (size < 100000 || !multithread) {
        alignas(64) uint32_t localHist[kNumSymbols] = {0};
        processBlock(in, size, localHist);
        
        for (int i = 0; i < kNumSymbols; i += 8) {
            _mm256_store_si256(
                reinterpret_cast<__m256i*>(out + i),
                _mm256_add_epi32(
                    _mm256_load_si256(reinterpret_cast<const __m256i*>(out + i)),
                    _mm256_load_si256(reinterpret_cast<const __m256i*>(localHist + i))
                )
            );
        }
        return;
    }

    const unsigned numThreads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;
    alignas(64) std::vector<uint32_t> histograms(numThreads * kNumSymbols, 0);

    const uint32_t blockSize = (size + numThreads * 4 - 1) / (numThreads * 4);
    std::atomic<uint32_t> currentBlock(0);

    for (unsigned t = 0; t < numThreads; ++t) {
        threads.emplace_back([&, t]() {
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            CPU_SET(t % numThreads, &cpuset);
            pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);

            uint32_t* localHist = &histograms[t * kNumSymbols];
            while (true) {
                const uint32_t blockIdx = currentBlock.fetch_add(1);
                const uint32_t start = blockIdx * blockSize;
                if (start >= size) break;
                const uint32_t end = std::min(start + blockSize, size);
                processBlock(in + start, end - start, localHist);
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }
    
    for (unsigned t = 0; t < numThreads; ++t) {
        const uint32_t* src = &histograms[t * kNumSymbols];
        #pragma omp simd aligned(src, out:64)
        for (int i = 0; i < kNumSymbols; ++i) {
            out[i] += src[i];
        }
    }
}

template <int one_bits, int kStateCheckMul>
void ansCalcWeights(
    int probBits,
    uint32_t totalNum,
    const uint32_t* counts,
    uint16_t* __restrict probsOut,
    uint4* table) {
    if (totalNum == 0) return;
    const uint32_t kProbWeight = 1 << probBits;
    int currentSum = 0;
    printf("totalNum: %d\n", totalNum);
    std::vector<uint32_t> qProb(kNumSymbols);
    std::vector<uint32_t> sortedPairs(kNumSymbols);
    for (int i = 0; i < kNumSymbols; ++i) {
        qProb[i] = static_cast<uint32_t>(kProbWeight * ((float)counts[i] / (float)(totalNum)));
        qProb[i] = (counts[i] > 0 && qProb[i] == 0) ? 1U : qProb[i];
        sortedPairs[i] = (qProb[i] << 16) | i;
        currentSum += qProb[i];
    }
    printf("currentSum: %d\n", currentSum);

    for(int i = 0; i < kNumSymbols; ++i)
    printf("i: %d, counts: %d\n", i, counts[i]);

    // #pragma omp single
    // {
    //     __gnu_parallel::sort(
    //         sortedPairs.begin(), 
    //         sortedPairs.end(),
    //         [](uint32_t a, uint32_t b) { return a > b; },
    //         __gnu_parallel::balanced_quicksort_tag()
    //     );
    // }
    std::sort(
      sortedPairs.begin(),
      sortedPairs.end(),
      [](uint32_t a, uint32_t b) { return a > b; }
    );

    uint32_t tidSymbol[kNumSymbols];
    for (int i = 0; i < kNumSymbols; ++i) {
        tidSymbol[i] = sortedPairs[i] & 0xFFFFU;
        qProb[i] = sortedPairs[i] >> 16;
    }
    
    for(int i = 0; i < kNumSymbols; ++i)
    printf("i: %d, tidSymbol: %d, qProb: %d\n", i, tidSymbol[i], qProb[i]);

    // #pragma omp parallel num_threads(32)
    // {
    //     int localSum = 0;
    //     #pragma omp for schedule(static)
    //     for (int i = 0; i < kNumSymbols; ++i) {
    //         localSum += qProb[i];
    //     }
    //     #pragma omp atomic
    //     currentSum += localSum;
    // }

    int diff = static_cast<int>(kProbWeight) - currentSum;

    if (diff > 0) {
      printf("diff: %d\n", diff);
      int iterToApply = std::min(diff, static_cast<int>(kNumSymbols));
      for(int i = diff; i > 0; i -= iterToApply){
        #pragma omp parallel for num_threads(32) schedule(static)
        for(int j = 0; j < kNumSymbols; ++j){
            qProb[j] += (tidSymbol[j] < iterToApply);       
        }
      }
    }
    else{
      printf("fucking\n");
      diff = -diff;
      while(diff > 0){
        int qNumGt1s = 0;
        for(int j = 0; j < kNumSymbols; ++j){
          qNumGt1s += (int)(qProb[j] > 1);
        }
        int iterToApply = diff < qNumGt1s ? diff : qNumGt1s;
        int startIndex = qNumGt1s - iterToApply;
        #pragma omp parallel for num_threads(32) schedule(static)
        for(int j = startIndex; j < qNumGt1s; ++j){
          qProb[j] --;
        }
        // for(int j = 0; j < kNumSymbols; ++j){
        //   if(j >= startIndex && j < qNumGt1s){
        //     qProb[j] -= 1;
        //   }
        // }
        diff -= iterToApply;
      }  
    }
    uint32_t symPdf[kNumSymbols];
    #pragma omp for simd schedule(static)
    for(int i = 0; i < kNumSymbols; i ++){
      symPdf[tidSymbol[i]] = qProb[i];
    }
    for(int i = 0; i < kNumSymbols; ++i)
    printf("i: %d, symPdf: %d\n", i, symPdf[i]);
    std::vector<uint16_t> cdf(kNumSymbols, 0);
    uint32_t pp = symPdf[0];
    probsOut[0] = pp;
    uint32_t shift0 = 32 - __builtin_clz(pp - 1);
    uint64_t magic0 = ((1ULL << 32) * ((1ULL << shift0) - pp)) / pp + 1;
    table[0] = {pp, 0, static_cast<uint32_t>(magic0), shift0
    };
    for (int i = 1; i < kNumSymbols; ++i) {
        uint32_t p = symPdf[i];
        probsOut[i] = p;
        uint32_t shift = 32 - __builtin_clz(p - 1);
        
        uint64_t magic = 0;
        if(p!= 0)
        magic = ((1ULL << 32) * ((1ULL << shift) - p)) / p + 1;
        cdf[i] = cdf[i-1] + symPdf[i-1];
        table[i] = {p, cdf[i], static_cast<uint32_t>(magic), shift};
        //, uint16_t(one_bits - p)
        // };
    }
}

template <int one_bits, int BlockSize, int kStateCheckMul>
void ansEncodeBatch(
    uint8_t* in,
    int inSize,
    uint32_t maxNumCompressedBlocks,
    uint32_t uncoalescedBlockStride,
    uint8_t* compressedBlocks_dev,
    uint32_t* compressedWords_dev,
    uint32_t* compressedWords_host_prefix,
    const uint4* table) {

    int num_threads = 16;
    #pragma omp parallel proc_bind(spread) num_threads(num_threads) 
    {
    // int thread_id = omp_get_thread_num();
    #pragma omp for schedule(dynamic, 8)
    for(int l = 0; l < maxNumCompressedBlocks; ++l){
    // for(int l = thread_id; l < maxNumCompressedBlocks; l += num_threads){
    uint32_t start = l * BlockSize;
    auto blockSize =  std::min(start + BlockSize, (uint32_t)inSize) - start;

    auto inBlock = in + start;
    auto outBlock = (ANSWarpState*)(compressedBlocks_dev
        + l * uncoalescedBlockStride);
    ANSEncodedT* outWords = (ANSEncodedT*)(outBlock + 1);
    uint64_t state[kWarpSize];
    std::fill(std::begin(state), std::end(state), kANSStartState);
    uint32_t outOffset = 0;
    uint32_t limit = roundDown(blockSize, 256);
    int cyclenum0 = limit >> 8;
    for (int i = 0; i < cyclenum0; ++i) {
      int idx0 = i << 8;
      for (int j = 0; j < 8; ++j) {
        int idx1 = idx0 + (j << 5);
        #pragma unroll(16)
        // #pragma omp simd
        for(int k = 0; k < kWarpSize; ++k){
          auto lookup = table[inBlock[k + idx1]];
          uint32_t pdf = lookup.x;
          uint32_t write_mask = (state[k] >= uint32_t(pdf << kStateCheckMul));
          outWords[outOffset] = (state[k] & kANSEncodedMask);
          outOffset += write_mask;
          state[k] >>= kANSEncodedBits * write_mask;
          uint64_t div = ((state[k] * lookup.z >> 32) + state[k]) >> lookup.w;
          state[k] += div * (one_bits - pdf) + (uint64_t)lookup.y;
        }
      }
    }
    
  if (blockSize - limit) {
    uint32_t limit1 = roundDown(blockSize, kWarpSize);
    int cyclenum1 = (limit1 - limit) / kWarpSize;
    for(int i = 0; i < cyclenum1; ++i){
      int idx = limit + (i << 5);
      for(int k = 0; k < kWarpSize; ++k){
          auto lookup = table[inBlock[k + idx]];
          uint64_t pdf = lookup.x;
          uint64_t tempstate = state[k];
          bool write_mask = (tempstate >= (pdf << kStateCheckMul));
          outWords[outOffset] = ((uint32_t)tempstate & kANSEncodedMask);
          outOffset += write_mask;
          tempstate >>= kANSEncodedBits * write_mask;
          uint64_t div = ((tempstate * lookup.z >> 32) + tempstate) >> lookup.w;
          state[k] = div * (one_bits - pdf) + (uint64_t)lookup.y + tempstate;
      }
    }
    if (blockSize - limit1) {
      int num = blockSize - limit1;
      for(int k = 0; k < num; ++k){
          auto lookup = table[inBlock[k + limit1]];
          uint64_t pdf = lookup.x;
          uint64_t tempstate = state[k];
          bool write_mask = (tempstate >= (pdf << kStateCheckMul));
          outWords[outOffset] = ((uint32_t)tempstate & kANSEncodedMask);
          outOffset += write_mask;
          tempstate >>= kANSEncodedBits * write_mask;
          uint64_t div = ((tempstate * lookup.z >> 32) + tempstate) >> lookup.w;
          state[k] = div * (one_bits - pdf) + (uint64_t)lookup.y + tempstate;
      }
    } 
  }
  auto outblockwarpstate = outBlock->warpState;
  #pragma omp simd
  for(int i = 0; i < kWarpSize; ++i){
    outblockwarpstate[i] = state[i];
  }
  compressedWords_dev[l] = outOffset;
  compressedWords_host_prefix[l] = roundUp(outOffset, kBlockAlignment / sizeof(ANSEncodedT));
  }
  }
}

void ansEncode(
    int precision,
    uint8_t* in,
    uint32_t inSize,
    uint8_t* out,
    uint32_t* outSize,
    ANSCoalescedHeader* headerOut,
    uint32_t& maxNumCompressedBlocks,
    uint32_t& uncoalescedBlockStride,
    uint8_t* compressedBlocks_host,
    uint32_t* compressedWords_host,
    uint32_t* compressedWords_host_prefix,
    uint32_t* compressedWordsPrefix_host) {
  ANSCoalescedHeader header;
  header.setProbBits(precision);

  uint32_t maxUncompressedWords = inSize / sizeof(ANSDecodedT);
  maxNumCompressedBlocks =
      (maxUncompressedWords + kDefaultBlockSize - 1) / kDefaultBlockSize;

  header.setNumBlocks(maxNumCompressedBlocks);
  header.setTotalUncompressedWords(inSize);

  uint4* table = (uint4*)malloc(sizeof(uint4) * kNumSymbols);
  uint32_t* tempHistogram = (uint32_t*)malloc(sizeof(uint32_t) * kNumSymbols);
  ansHistogram(
      in,
      inSize,
      tempHistogram);

#define RUN_ENCODE(ONEBITS, kStateCheckMul)                                       \
  do {    \
    ansCalcWeights<ONEBITS, kStateCheckMul>(\
      precision,\
      inSize,\
      tempHistogram,\
      headerOut->getSymbolProbs(),\
      table);  \
        printf("1\n");                                                   \
    ansEncodeBatch<ONEBITS, kDefaultBlockSize, kStateCheckMul>(\
            in,\
            inSize,                                        \
            maxNumCompressedBlocks,                            \
            uncoalescedBlockStride,                            \
            compressedBlocks_host,                       \
            compressedWords_host,  \
            compressedWords_host_prefix,                      \
            table);                                 \
  } while (false)

    switch (precision) {
      case 9:
        RUN_ENCODE(512, 22);
        break;
      case 10:
        RUN_ENCODE(1024, 21);
        break;
      case 11:
        RUN_ENCODE(2048, 20);
        break;
      default:
        std::cout<< "unhandled pdf precision " << precision << std::endl;
    }

#undef RUN_ENCODE
  uint32_t totalCompressedWords = 0;
  if(maxNumCompressedBlocks > 0){
    std::exclusive_scan(compressedWords_host_prefix, compressedWords_host_prefix + maxNumCompressedBlocks, compressedWordsPrefix_host, 0);
    totalCompressedWords =
        compressedWordsPrefix_host[maxNumCompressedBlocks - 1] +
            roundUp(
            compressedWords_host[maxNumCompressedBlocks - 1],
            kBlockAlignment / sizeof(ANSEncodedT));
  }
  header.setTotalCompressedWords(totalCompressedWords);

  *outSize = header.getTotalCompressedSize();
  *headerOut = header;
}
} // namespace 

#undef RUN_ENCODE_ALL

#endif

