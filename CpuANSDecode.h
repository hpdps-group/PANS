#ifndef CPU_ANS_INCLUDE_ANS_CPUANSDECODE_H
#define CPU_ANS_INCLUDE_ANS_CPUANSDECODE_H

#pragma once

#include "CpuANSUtils.h"

namespace cpu_ans {

// inline uint32_t packDecodeLookup(uint32_t sym, uint32_t pdf, uint32_t cdf) {
//   // [31:20] cdf
//   // [19:8] pdf
//   // [7:0] symbol
//   return (cdf << 20) | (pdf << 8) | sym;
// }

// inline void unpackDecodeLookup(uint32_t v, uint32_t& sym, uint32_t& pdf, uint32_t& cdf) {
//   // [31:20] cdf
//   // [19:8] pdf
//   // [7:0] symbol
//   sym = v & 0xffU;
//   v >>= 8;
//   pdf = v & 0xfffU;
//   v >>= 12;
//   cdf = v;
// }

template <int ProbBits,
    int kDefaultBlockSize>
void ansDecodeKernel_opti(
    void* in,
    void* out
    ) {
  int num_threads = 16;
  auto headerIn = (ANSCoalescedHeader*)in;
  auto opdf = headerIn->getSymbolProbs();
  std::vector<uint32_t> ocdf(kNumSymbols);
  std::exclusive_scan(opdf, opdf + kNumSymbols, ocdf.begin(), 0);
  uint32_t* symbol = (uint32_t*)std::aligned_alloc(kBlockAlignment, sizeof(uint32_t) * (1 << ProbBits));
  uint32_t* pdf = (uint32_t*)std::aligned_alloc(kBlockAlignment, sizeof(uint32_t) * (1 << ProbBits));
  uint32_t* cdf = (uint32_t*)std::aligned_alloc(kBlockAlignment, sizeof(uint32_t) * (1 << ProbBits));
  #pragma unroll 4
  for(uint32_t i = 0; i < kNumSymbols; i ++){
    auto smempdf = opdf[i];
    auto begin = ocdf[i];
    for(int j = begin, k = 0; j < begin + smempdf; j ++, k ++){
        symbol[j] = i;
        pdf[j] = smempdf;
        cdf[j] = k;
    }
  }
  auto header = *headerIn;
  auto numBlocks = header.getNumBlocks();
  auto totalUncompressedWords = header.getTotalUncompressedWords();
  constexpr ANSStateT StateMask = (ANSStateT(1) << ProbBits) - ANSStateT(1);
  
  #pragma omp parallel proc_bind(spread) num_threads(num_threads)
  {
    int thread_id = omp_get_thread_num();
    // #pragma omp for schedule(dynamic, 8)
    // #pragma omp for 
    for(int i = thread_id; i < numBlocks; i += num_threads){
    
    // for(int i = 0; i < numBlocks; i ++){
      ANSStateT state[kWarpSize];
      #pragma unroll
      for(int j = 0; j < kWarpSize; j ++){
          state[j] = ANSStateT(headerIn->getWarpStates()[i].warpState[j]);
      }
      auto blockWords = headerIn->getBlockWords(numBlocks)[i];
      uint32_t uncompressedWords = (blockWords.x >> 16);
      uint32_t compressedWords = (blockWords.x & 0xffff);
      uint32_t blockCompressedWordStart = blockWords.y;
      ANSEncodedT* blockDataIn =
          headerIn->getBlockDataStart(numBlocks) + blockCompressedWordStart;
      uint8_t* outBlock_ = (uint8_t*)out + (i << 12);
      if(uncompressedWords == kDefaultBlockSize){
          #pragma unroll
          for(int k = kDefaultBlockSize - kWarpSize; k >= 0; k -= kWarpSize){
              for(int j = kWarpSize - 1; j >= 0; j --){ 
                auto s_bar = state[j] & StateMask;
                state[j] = pdf[s_bar] * (state[j] >> ProbBits) + ANSStateT(cdf[s_bar]);
                bool read = state[j] < kANSMinState;
                if(read){
                    compressedWords--;
                    auto v = blockDataIn[compressedWords];
                    state[j] = ((state[j] << kANSEncodedBits) + ANSStateT(v));
                }
                outBlock_[k + j] = symbol[s_bar];
              }
          }
      } 
      else {
          uint32_t remainder = uncompressedWords & 31;
          int uncompressedOffset = uncompressedWords - remainder;
          if(remainder > 0){
              for(int j = kWarpSize - 1; j >= 0; j --){
                  bool valid = j < remainder;
                  auto s_bar = state[j] & StateMask;
                  if(valid){
                      state[j] = pdf[s_bar] * (state[j] >> ProbBits) + ANSStateT(cdf[s_bar]);
                  }
                  bool read = valid && (state[j] < kANSMinState);
                  if(read){
                      compressedWords--;
                      auto v = blockDataIn[compressedWords];
                      state[j] = ((state[j] << kANSEncodedBits) + ANSStateT(v));
                  }
                  if(valid){
                      outBlock_[uncompressedOffset + j] = symbol[s_bar];
                  }
              }
          }
          while(uncompressedOffset > 0){
              uncompressedOffset -= kWarpSize;
              for(int j = kWarpSize - 1; j >= 0; j --){
                  auto s_bar = state[j] & StateMask;
                  state[j] = pdf[s_bar] * (state[j] >> ProbBits) + ANSStateT(cdf[s_bar]);
                  bool read = state[j] < kANSMinState;
                  if(read){
                      compressedWords--;
                      auto v = blockDataIn[compressedWords];
                      state[j] = ((state[j] << kANSEncodedBits) + ANSStateT(v));
                  }
                  outBlock_[uncompressedOffset + j] = symbol[s_bar];
              }
          }
      }
    }
  }
}

void ansDecode(
    int precision,
    uint8_t* in,
    uint8_t* out
    ) {
  
  {
#define RUN_DECODE(BITS)                                           \
  do { ansDecodeKernel_opti<BITS, kDefaultBlockSize>(in, out);} while (false)   \
    
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
}
} // namespace 

#endif