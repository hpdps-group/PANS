/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef CPU_ANS_INCLUDE_ANS_CPUANSDECODE_H
#define CPU_ANS_INCLUDE_ANS_CPUANSDECODE_H

#pragma once

#include <omp.h>
#include <cmath>
#include <memory>
#include <sstream>
#include <vector>
#include "CpuANSUtils.h"

namespace cpu_ans {

uint32_t packDecodeLookup(uint32_t sym, uint32_t pdf, uint32_t cdf) {return (cdf << 20) | (pdf << 8) | sym;}

void unpackDecodeLookup(uint32_t v, uint32_t& sym, uint32_t& pdf, uint32_t& cdf) {
  sym = v & 0xffU;
  v >>= 8;
  pdf = v & 0xfffU;
  v >>= 12;
  cdf = v;
}

template <int ProbBits,
    int kDefaultBlockSize>
void ansDecodeKernel(
    void* in,
    uint32_t* table,
    void* out
    ) {
  auto headerIn = (ANSCoalescedHeader*)in;
  headerIn->checkMagicAndVersion();
  auto header = *headerIn;
  auto numBlocks = header.getNumBlocks();
  auto totalUncompressedWords = header.getTotalUncompressedWords();
  auto writer = BatchWriter(out);
  for(int i = 0; i < numBlocks; i ++){
    ANSStateT state[kWarpSize];
    for(int j = 0; j < kWarpSize; j ++){
        state[j] = ANSStateT(headerIn->getWarpStates()[i].warpState[j]);
    }
    auto blockWords = headerIn->getBlockWords(numBlocks)[i];
    uint32_t uncompressedWords = (blockWords.x >> 16);
    uint32_t compressedWords = (blockWords.x & 0xffff);
    uint32_t blockCompressedWordStart = blockWords.y;
    ANSEncodedT* blockDataIn =
        headerIn->getBlockDataStart(numBlocks) + blockCompressedWordStart;
    writer.setBlock(i);
    if(uncompressedWords == kDefaultBlockSize){
        for(int k = kDefaultBlockSize - kWarpSize; k >= 0; k -= kWarpSize){
            int count = 0;
            for(int j = kWarpSize - 1; j >= 0; j --){
                constexpr ANSStateT StateMask = (ANSStateT(1) << ProbBits) - ANSStateT(1);
                auto s_bar = state[j] & StateMask;
                uint32_t sym; 
                uint32_t pdf;
                uint32_t sMinusCdf;
                unpackDecodeLookup(table[s_bar], sym, pdf, sMinusCdf);
                auto pre_state = state[j];
                state[j] = pdf * (state[j] >> ProbBits) + ANSStateT(sMinusCdf);
                bool read = state[j] < kANSMinState;
                if(read){
                    count++;
                    auto v = blockDataIn[compressedWords - count];
                    if(compressedWords - count < 0)
                    printf("fucking error");
                    state[j] = ((state[j] << kANSEncodedBits) + ANSStateT(v));
                }
                writer.write(k + j, sym);             
            }
            compressedWords -= count;
        }
    } 
    else {
        uint32_t remainder = uncompressedWords % kWarpSize;
        int uncompressedOffset = uncompressedWords - remainder;
        uint32_t compressedOffset = compressedWords;
        if(remainder > 0){
            uint32_t numCompressedRead = 0;
            for(int j = kWarpSize - 1; j >= 0; j --){
                bool valid = j < remainder;
                constexpr ANSStateT StateMask = (ANSStateT(1) << ProbBits) - ANSStateT(1);
                auto s_bar = state[j] & StateMask;  
                uint32_t sym;
                uint32_t pdf;
                uint32_t sMinusCdf;
                unpackDecodeLookup(table[s_bar], sym, pdf, sMinusCdf);
                if(valid){
                    state[j] = pdf * (state[j] >> ProbBits) + ANSStateT(sMinusCdf);
                }
                bool read = valid && (state[j] < kANSMinState);
                if(read){
                    numCompressedRead++;
                    auto v = blockDataIn[compressedOffset - numCompressedRead];
                    state[j] = ((state[j] << kANSEncodedBits) + ANSStateT(v));
                }
                if(valid){
                    writer.write(uncompressedOffset + j, sym);
                }
            }
            compressedOffset -= numCompressedRead;
        }
        while(uncompressedOffset > 0){
            uncompressedOffset -= kWarpSize;
            uint32_t numCompressedRead = 0;
            for(int j = kWarpSize - 1; j >= 0; j --){
                constexpr ANSStateT StateMask = (ANSStateT(1) << ProbBits) - ANSStateT(1);
                auto s_bar = state[j] & StateMask;
                uint32_t sym;
                uint32_t pdf;
                uint32_t sMinusCdf;
                unpackDecodeLookup(table[s_bar], sym, pdf, sMinusCdf);
                state[j] = pdf * (state[j] >> ProbBits) + ANSStateT(sMinusCdf);
                bool read = state[j] < kANSMinState;
                if(read){
                    numCompressedRead++;
                    auto v = blockDataIn[compressedOffset - numCompressedRead];
                    state[j] = ((state[j] << kANSEncodedBits) + ANSStateT(v));
                }
                writer.write(uncompressedOffset + j, sym);
            }
            compressedOffset -= numCompressedRead;
        }
    }
  }
}

std::vector<uint32_t> parallelPrefixSum(const std::vector<uint32_t>& pdf) {
    std::vector<uint32_t> cdf(pdf.size());
    
    #pragma omp parallel
    {
        #pragma omp for schedule(static)
        for (int i = 0; i < pdf.size(); ++i) {
            cdf[i] = (i == 0) ? 0 : pdf[i-1];
        }

        #pragma omp single
        {
            for (int i = 1; i < pdf.size(); ++i) {
                cdf[i] += cdf[i-1];
            }
        }
    }
    
    return cdf;
}

void ansDecodeTable(
    void* in,
    uint32_t probBits,
    uint32_t* __restrict__ table) {
  auto headerIn = (ANSCoalescedHeader*)in;
  auto header = *headerIn;
  if (header.getTotalUncompressedWords() == 0) {return;}
  auto probs = headerIn->getSymbolProbs();
  std::vector<uint32_t> pdf(kNumSymbols);
  for(int i = 0; i < kNumSymbols; i ++)
    pdf[i] = probs[i];
  std::vector<uint32_t> cdf = parallelPrefixSum(pdf);
  uint2 smemPdfCdf[kNumSymbols];
  for(int i = 0; i < kNumSymbols; i ++) {
    smemPdfCdf[i] = {pdf[i], cdf[i]};
  }
  for(int i = 0; i < kNumSymbols; i ++){
    auto v = smemPdfCdf[i];
    auto smempdf = v.x;
    auto begin = v.y;
    auto end = begin + smempdf;
    for(int j = begin; j < end; j ++){
        table[j] = packDecodeLookup(
            i, 
            smempdf, 
            j - begin); 
    }
  }
}

void ansDecode(
    int precision,
    uint8_t* in,
    uint8_t* out
    ) {
  uint32_t* table = new uint32_t[1 << precision];
  ansDecodeTable(in, precision, table);
  {
#define RUN_DECODE(BITS)                                           \
  do { ansDecodeKernel<BITS, kDefaultBlockSize>(in, table, out);} while (false)   \
    
    switch (precision) {
      case 9:
        RUN_DECODE(9);
        break;
      case 10:
        RUN_DECODE(10);
        break;
      case 11:
        RUN_DECODE(11);
        break;
      default:
        std::cout << "unhandled pdf precision " << precision << std::endl;
    }
#undef RUN_DECODE
  }
  delete[] table;
}
} // namespace 

#endif