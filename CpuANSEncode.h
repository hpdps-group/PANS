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
    // 处理未对齐的前缀部分
    uint32_t roundUp = std::min(size, static_cast<uint32_t>(getAlignmentRoundUp(kAlign, in)));
    for (uint32_t i = 0; i < roundUp; ++i) {
        ++localHist[in[i]];  // 使用前置递增减少依赖
    }

    const uint8_t* alignedIn = in + roundUp;
    uint32_t remaining = size - roundUp;
    uint32_t numChunks = remaining / kAlign;

    // 处理对齐的32字节块
    const __m256i* avxIn = reinterpret_cast<const __m256i*>(alignedIn);
    for (uint32_t i = 0; i < numChunks; ++i) {
        const __m256i vec = _mm256_load_si256(avxIn + i);
        const uint8_t* bytes = reinterpret_cast<const uint8_t*>(avxIn + i);
        
        // 分组处理并插入软件预取
        _mm_prefetch(reinterpret_cast<const char*>(avxIn + i + 1), _MM_HINT_T0);
        
        // 使用双寄存器流水线处理
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

    // 用switch处理剩余0-7字节
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
        
        // 向量化合并
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

    // 动态任务分块（避免尾部不均）
    const uint32_t blockSize = (size + numThreads * 4 - 1) / (numThreads * 4);
    std::atomic<uint32_t> currentBlock(0);

    for (unsigned t = 0; t < numThreads; ++t) {
        threads.emplace_back([&, t]() {
            // 设置CPU亲和性
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
    
    // 优化合并：按线程顺序累加，提升缓存效率
    for (unsigned t = 0; t < numThreads; ++t) {
        const uint32_t* src = &histograms[t * kNumSymbols];
        #pragma omp simd aligned(src, out:64)
        for (int i = 0; i < kNumSymbols; ++i) {
            out[i] += src[i];
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
    else{
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
    // constexpr ANSStateT kStateCheckMul = kANSStateBits - ProbBits;
    int num_threads = 16;
    #pragma omp parallel proc_bind(spread) num_threads(num_threads) 
    {
    int thread_id = omp_get_thread_num();
    // #pragma omp for schedule(dynamic, 8)
    // for(int l = 0; l < maxNumCompressedBlocks; ++l){
    for(int l = thread_id; l < maxNumCompressedBlocks; l += num_threads){
    uint32_t start = l * BlockSize;
    auto blockSize =  std::min(start + BlockSize, (uint32_t)inSize) - start;
    auto inBlock = in + start;
    auto outBlock = (ANSWarpState*)(compressedBlocks_dev
        + l * uncoalescedBlockStride);
    ANSEncodedT* outWords = (ANSEncodedT*)(outBlock + 1);
    // uint32_t inOffset[kWarpSize];
    uint32_t state[kWarpSize];// = {kANSStartState};
    std::fill(std::begin(state), std::end(state), kANSStartState);
    // #pragma unroll(4)
    // for(int i = 0; i < kWarpSize; ++i){
    //   // inOffset[i] = i;
    //   state[i] = kANSStartState;
    // }

    uint32_t outOffset = 0;
    uint32_t limit = roundDown(blockSize, 256);
    int cyclenum0 = limit >> 8;
    for (int i = 0; i < cyclenum0; ++i) {
      for (int j = 0; j < 8; ++j) {
        int idx = (i << 8) + (j << 5);
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
    }
  if (blockSize - limit) {
    uint32_t limit1 = roundDown(blockSize, kWarpSize);
    int cyclenum1 = (limit1 - limit) / kWarpSize;
    for(int i = 0; i < cyclenum1; ++i){
      int idx = limit + (i << 5);
      for(int k = 0; k < kWarpSize; ++k){
          // auto lookup = table[inBlock[inOffset[k]]];
          auto lookup = table[inBlock[k + idx]];
          // uint32_t pdf = lookup.x;
          // ANSStateT maxStateCheck = pdf << kStateCheckMul;
          // uint32_t write = (state[k] >= maxStateCheck);
          // if (write) {
          //   outWords[outOffset] = (state[k] & kANSEncodedMask);
          //   state[k] = (state[k] >> kANSEncodedBits);
          //   outOffset ++;
          // }
          // uint32_t div = (umulhi(state[k], lookup.z) + state[k]) >> lookup.w;
          // state[k] = (div * one_bits) + state[k] - (div * pdf) + lookup.y;
          // inOffset[k] += kWarpSize;
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
          // auto lookup = table[inBlock[inOffset[k]]];
          auto lookup = table[inBlock[k + limit1]];
          // uint32_t pdf = lookup.x;
          // ANSStateT maxStateCheck = pdf << kStateCheckMul;
          // uint32_t write_mask = (state[k] >= maxStateCheck);
          // // if (write_mask) {
          // //   outWords[outOffset] = (state[k] & kANSEncodedMask);
          // //   state[k] = (state[k] >> kANSEncodedBits);
          // //   outOffset ++;
          // // }
          // outWords[outOffset] = (state[k] & kANSEncodedMask) * write_mask;
          // outOffset += (write_mask != 0);
          // state[k] = (state[k] >> kANSEncodedBits) * write_mask + state[k] * (!write_mask);
          // uint32_t div = (umulhi(state[k], lookup.z) + state[k]) >> lookup.w;
          // state[k] = (div * one_bits) + state[k] - (div * pdf) + lookup.y;
          uint64_t pdf = lookup.x;
          uint64_t tempstate = state[k];
          bool write_mask = (tempstate >= (pdf << kStateCheckMul));
          outWords[outOffset] = ((uint32_t)tempstate & kANSEncodedMask);
          // if(i == 15 && j == 7 && l ==12287)
          // printf("k:%d, idx:%d, pdf:%d, tempstate:%d, inBLock[k + idx]:%d, lookup.x:%d, lookup.y:%d, lookup.z:%d, lookup.w:%d, write_mask:%d, outOffset:%d, outWords[outOffset]:%d\n", k, idx, pdf, tempstate, inBlock[k + idx], lookup.x, lookup.y, lookup.z, lookup.w, write_mask, outOffset, outWords[outOffset]);
          outOffset += write_mask;
          tempstate >>= kANSEncodedBits * write_mask;
          uint64_t div = ((tempstate * lookup.z >> 32) + tempstate) >> lookup.w;
          state[k] = div * (one_bits - pdf) + (uint64_t)lookup.y + tempstate;
      }
    } 
  }
  #pragma omp simd
  for(int i = 0; i < kWarpSize; ++i){
    outBlock->warpState[i] = state[i];
  }
  compressedWords_dev[l] = outOffset;
  compressedWords_host_prefix[l] = roundUp(outOffset, kBlockAlignment / sizeof(ANSEncodedT));
  }
  }
}


void ansEncodeCoalesceBatch(
    const uint8_t* __restrict__ compressedBlocks_host,
    int uncompressedWords,
    uint32_t maxNumCompressedBlocks,
    uint32_t uncoalescedBlockStride,
    const uint32_t* __restrict__ compressedWords_host,
    const uint32_t* __restrict__ compressedWords_host_prefix,
    uint32_t* __restrict__ compressedWordsPrefix_host,
    const uint4* __restrict__ table,
    uint32_t config_probBits,
    uint8_t* out,
    uint32_t* outSize) {

  ANSCoalescedHeader* headerOut = (ANSCoalescedHeader*)out;
  uint32_t totalCompressedWords = 0;
  // std::exclusive_scan(compressedWords_host, compressedWords_host + maxNumCompressedBlocks, compressedWordsPrefix_host, 0);
  if(maxNumCompressedBlocks > 0){
    std::exclusive_scan(compressedWords_host_prefix, compressedWords_host_prefix + maxNumCompressedBlocks, compressedWordsPrefix_host, 0);
    totalCompressedWords =
        compressedWordsPrefix_host[maxNumCompressedBlocks - 1] +
            roundUp(
            compressedWords_host[maxNumCompressedBlocks - 1],
            kBlockAlignment / sizeof(ANSEncodedT));
  }
    
  ANSCoalescedHeader header;
  header.setMagicAndVersion();
  header.setNumBlocks(maxNumCompressedBlocks);
  // printf("maxNumCompressedBlocks:%d\n",maxNumCompressedBlocks);
  header.setTotalUncompressedWords(uncompressedWords);
  // printf("uncompressedWords:%d\n",uncompressedWords);
  header.setTotalCompressedWords(totalCompressedWords);
  // printf("totalCompressedWords:%d\n",totalCompressedWords);
  header.setProbBits(config_probBits);

  *outSize = header.getTotalCompressedSize();
  *headerOut = header;

  auto probsOut = headerOut->getSymbolProbs();
  
  // Write out pdf
  for (int j = 0; j < kNumSymbols; j ++) {
    probsOut[j] = table[j].x;
  }
  auto blockWordsOut = headerOut->getBlockWords(maxNumCompressedBlocks);
  auto BlockDataStart = headerOut->getBlockDataStart(maxNumCompressedBlocks);

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

    // memcpy(outT, inT, limitEnd << 4);
    __builtin_memcpy(outT, inT, limitEnd << 4);
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

  // memcpy(outT, inT, limitEnd << 4);
  __builtin_memcpy(outT, inT, limitEnd << 4);
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
  // auto start = std::chrono::high_resolution_clock::now();
  ansHistogram(
      in,
      inSize,
      tempHistogram);
//   auto end = std::chrono::high_resolution_clock::now();
//     // 计算运行时间
//     auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

//     std::cout << "ansHistogram: " << duration.count() / 1000.0 << " ms" << std::endl;

// start = std::chrono::high_resolution_clock::now();
  ansCalcWeights(
      precision,
      inSize,
      tempHistogram,
      table);
    // end = std::chrono::high_resolution_clock::now();

    // // 计算运行时间
    // duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // std::cout << "ansCalcWeights: " << duration.count() / 1000.0 << " ms" << std::endl;
  
  // for(int i = 0; i < kNumSymbols; i ++){
  //   printf("table[%d] = {%d, %d, %d, %d}\n", i, table[i].x, table[i].y, table[i].z, table[i].w);
  // }
  // void* aligned_memory = std::aligned_alloc(ALIGNMENT, size);
  uint32_t uncoalescedBlockStride = getMaxBlockSizeUnCoalesced(kDefaultBlockSize);
  // uint8_t* compressedBlocks_host = (uint8_t*)malloc(sizeof(uint8_t) * maxNumCompressedBlocks * uncoalescedBlockStride);
  // uint32_t* compressedWords_host = (uint32_t*)malloc(sizeof(uint32_t) * maxNumCompressedBlocks);
  // uint32_t* compressedWordsPrefix_host = (uint32_t*)malloc(sizeof(uint32_t) * maxNumCompressedBlocks);
  uint8_t* compressedBlocks_host = (uint8_t*)std::aligned_alloc(kBlockAlignment, sizeof(uint8_t) * maxNumCompressedBlocks * uncoalescedBlockStride);
  uint32_t* compressedWords_host = (uint32_t*)std::aligned_alloc(kBlockAlignment, sizeof(uint32_t) * maxNumCompressedBlocks);
  uint32_t* compressedWords_host_prefix = (uint32_t*)std::aligned_alloc(kBlockAlignment, sizeof(uint32_t) * maxNumCompressedBlocks);
  uint32_t* compressedWordsPrefix_host = (uint32_t*)std::aligned_alloc(kBlockAlignment, sizeof(uint32_t) * maxNumCompressedBlocks);
// start = std::chrono::high_resolution_clock::now();
#define RUN_ENCODE(ONEBITS, kStateCheckMul)                                       \
  do {                                                         \
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
    // end = std::chrono::high_resolution_clock::now();

    // // 计算运行时间
    // duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // std::cout << "ansEncodeBatch: " << duration.count() / 1000.0 << " ms" << std::endl;

  // start = std::chrono::high_resolution_clock::now();
  // std::exclusive_scan(compressedWords_host, compressedWords_host + maxNumCompressedBlocks, compressedWordsPrefix_host, 0);
  // if (maxNumCompressedBlocks > 0) {
  //   compressedWordsPrefix_host[0] = 0;
  //   for(int i = 1; i < maxNumCompressedBlocks; i ++){
  //     compressedWordsPrefix_host[i] = compressedWordsPrefix_host[i - 1] + roundUp(compressedWords_host[i - 1], kBlockAlignment / sizeof(ANSEncodedT));
  //   }
  // } 
  // end = std::chrono::high_resolution_clock::now();
  // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

  // std::cout << "exclusive_scan: " << duration.count() / 1000.0 << " ms" << std::endl;
  // for(int i = 0; i < 10; i ++){
  //   printf("compressedWords_host[%d] = %d\n", i, compressedWords_host[i]);
  // } 
  // for(int i = 0; i < 10; i ++){
  //   printf("compressedWordsPrefix_host[%d] = %d\n", i, compressedWordsPrefix_host[i]);
  // } 
  // for(int i = 0; i < maxNumCompressedBlocks; i ++){
  //   printf("compressedWords_host[%d] = %d\n", i, compressedWords_host[i]);
  // }
  // for(int i = 0; i < maxNumCompressedBlocks; i ++){
  // start = std::chrono::high_resolution_clock::now();
  ansEncodeCoalesceBatch(
          compressedBlocks_host,
          inSize,
          maxNumCompressedBlocks,
          uncoalescedBlockStride,
          compressedWords_host,
          compressedWords_host_prefix,
          compressedWordsPrefix_host,
          table,
          precision,
          out,
          outSize);
    // printf("compressedWordsPrefix_host[12289] = %d\n", compressedWordsPrefix_host[12289]);
    // end = std::chrono::high_resolution_clock::now();

    // // 计算运行时间
    // duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // std::cout << "ansEncodeCoalesceBatch: " << duration.count() / 1000.0 << " ms" << std::endl;

}
} // namespace 

#undef RUN_ENCODE_ALL

#endif

