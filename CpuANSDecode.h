#ifndef CPU_ANS_INCLUDE_ANS_CPUANSDECODE_H
#define CPU_ANS_INCLUDE_ANS_CPUANSDECODE_H

#pragma once

#include "CpuANSUtils.h"

namespace cpu_ans {

inline uint32_t packDecodeLookup(uint32_t sym, uint32_t pdf, uint32_t cdf) {
  // [31:20] cdf
  // [19:8] pdf
  // [7:0] symbol
  return (cdf << 20) | (pdf << 8) | sym;
}

inline void unpackDecodeLookup(uint32_t v, uint32_t& sym, uint32_t& pdf, uint32_t& cdf) {
  // [31:20] cdf
  // [19:8] pdf
  // [7:0] symbol
  sym = v & 0xffU;
  v >>= 8;
  pdf = v & 0xfffU;
  v >>= 12;
  cdf = v;
}

template <int ProbBits,
    int kDefaultBlockSize>
void ansDecodeKernel_opti(
    void* in,
    void* out
    ) {
  int num_threads = 16;
  auto headerIn = (ANSCoalescedHeader*)in;
  auto opdf = headerIn->getSymbolProbs();
  // __builtin_prefetch(opdf, 0, 3);
  std::vector<uint32_t> ocdf(kNumSymbols);
  std::exclusive_scan(opdf, opdf + kNumSymbols, ocdf.begin(), 0);
  uint32_t* symbol = (uint32_t*)std::aligned_alloc(kBlockAlignment, sizeof(uint32_t) * (1 << ProbBits));
  uint32_t* pdf = (uint32_t*)std::aligned_alloc(kBlockAlignment, sizeof(uint32_t) * (1 << ProbBits));
  uint32_t* cdf = (uint32_t*)std::aligned_alloc(kBlockAlignment, sizeof(uint32_t) * (1 << ProbBits));
  // uint32_t* table_dev = (uint32_t*)std::aligned_alloc(kBlockAlignment, sizeof(uint32_t) * (1 << ProbBits) * 3);
  #pragma unroll 
  // #pragma omp parallel for 
  for(uint32_t i = 0; i < kNumSymbols; i ++){
    auto smempdf = opdf[i];
    auto begin = ocdf[i];
    auto end = begin + smempdf;
    for(int j = begin, k = 0; j < end; j ++, k ++){
        symbol[j] = i;
        pdf[j] = smempdf;
        cdf[j] = k;
    }
  }
  auto header = *headerIn;
  auto numBlocks = header.getNumBlocks();
  auto totalUncompressedWords = header.getTotalUncompressedWords();
  constexpr ANSStateT StateMask = (ANSStateT(1) << ProbBits) - ANSStateT(1);
  auto blockWordspre = headerIn->getBlockWords(numBlocks);
  auto blockDataInStart = headerIn->getBlockDataStart(numBlocks);

  // __builtin_prefetch(headerIn->getWarpStates()[0].warpState, 0, 0);

  #pragma omp parallel proc_bind(spread) num_threads(num_threads)
  {
    __builtin_prefetch(headerIn->getWarpStates(), 0, 0);
    __builtin_prefetch(blockWordspre, 0, 0);
    __builtin_prefetch(blockDataInStart, 0, 0);
    // __builtin_prefetch(symbol, 0, 0);
    // __builtin_prefetch(pdf, 0, 0);
    // __builtin_prefetch(cdf, 0, 0);
    int thread_id = omp_get_thread_num();
    #pragma omp for schedule(dynamic, 8)
    // #pragma omp for 
    // for(int i = thread_id; i < numBlocks; i += num_threads){
    for(int i = 0; i < numBlocks; i ++){
      ANSStateT state[kWarpSize];
      auto State = headerIn->getWarpStates()[i].warpState;
      #pragma  unroll 16
      for(int j = 0; j < kWarpSize; j ++){
          state[j] = State[j];
      }
      auto blockWords = blockWordspre[i];
      uint32_t uncompressedWords = (blockWords.x >> 16);
      uint32_t compressedWords = (blockWords.x & 0xffff);
      uint32_t blockCompressedWordStart = blockWords.y;
      ANSEncodedT* blockDataIn =
          blockDataInStart + blockCompressedWordStart;
      __builtin_prefetch(blockDataIn, 0, 0);
      uint8_t* outBlock_ = (uint8_t*)out + (i << 12);
      if(uncompressedWords == kDefaultBlockSize){
          // #pragma unroll
          // #pragma omp simd
          for(int k = kDefaultBlockSize - kWarpSize; k >= 0; k -= kWarpSize){
            // uint32_t outsym[kWarpSize];
              // for(int j = kWarpSize - 1; j >= 0; j --){ 
              //   auto s_bar = state[j] & StateMask;
              //   state[j] = pdf[s_bar] * (state[j] >> ProbBits) + ANSStateT(cdf[s_bar]);
              //   uint32_t read = state[j] < kANSMinState;
              //   compressedWords -= read;
              //   auto v = blockDataIn[compressedWords];
              //   state[j] = ((state[j] << (kANSEncodedBits * read)) + ANSStateT(v) * read);
              //   outBlock_[k + j] = symbol[s_bar];
              // }
            uint64_t outsym;
            uint32_t s_bar;
            uint32_t read;
            uint16_t v;
            int tempk = k >> 3;
            uint8_t tempoutsym[8];
            // __builtin_prefetch(blockDataIn + compressedWords, 0, 0);
            for(int j = kWarpSize - 8, l = 3; j >= 0; j -= 8, l --){
              int temp = j + 7;
              s_bar = state[temp] & StateMask;
              state[temp] = pdf[s_bar] * (state[temp] >> ProbBits) + ANSStateT(cdf[s_bar]);
              read = state[temp] < kANSMinState;
              compressedWords -= read;
              v = blockDataIn[compressedWords];
              state[temp] = ((state[temp] << (kANSEncodedBits * read)) + ANSStateT(v) * read);
              // outsym = (outsym << 8) | symbol[s_bar];
              // outBlock_[k + j] = symbol[s_bar];
              tempoutsym[7] = symbol[s_bar];

              temp --;
              s_bar = state[temp] & StateMask;
              state[temp] = pdf[s_bar] * (state[temp] >> ProbBits) + ANSStateT(cdf[s_bar]);
              read = state[temp] < kANSMinState;
              compressedWords -= read;
              v = blockDataIn[compressedWords];
              state[temp] = ((state[temp] << (kANSEncodedBits * read)) + ANSStateT(v) * read);
              // outsym = (outsym << 8) | symbol[s_bar];
              // outBlock_[k + j1] = symbol[s_bar];
              tempoutsym[6] = symbol[s_bar];

              temp --;
              s_bar = state[temp] & StateMask;
              state[temp] = pdf[s_bar] * (state[temp] >> ProbBits) + ANSStateT(cdf[s_bar]);
              read = state[temp] < kANSMinState;
              compressedWords -= read;
              v = blockDataIn[compressedWords];
              state[temp] = ((state[temp] << (kANSEncodedBits * read)) + ANSStateT(v) * read);
              // outsym = (outsym << 8) | symbol[s_bar];
              // outBlock_[k + j2] = symbol[s_bar];
              tempoutsym[5] = symbol[s_bar];

              temp --;
              s_bar = state[temp] & StateMask;
              state[temp] = pdf[s_bar] * (state[temp] >> ProbBits) + ANSStateT(cdf[s_bar]);
              read = state[temp] < kANSMinState;
              compressedWords -= read;
              v = blockDataIn[compressedWords];
              state[temp] = ((state[temp] << (kANSEncodedBits * read)) + ANSStateT(v) * read);
              // outsym = (outsym << 8) | symbol[s_bar];
              // outBlock_[k + j3] = symbol[s_bar];
              tempoutsym[4] = symbol[s_bar];

              temp --;
              s_bar = state[temp] & StateMask;
              state[temp] = pdf[s_bar] * (state[temp] >> ProbBits) + ANSStateT(cdf[s_bar]);
              read = state[temp] < kANSMinState;
              compressedWords -= read;
              v = blockDataIn[compressedWords];
              state[temp] = ((state[temp] << (kANSEncodedBits * read)) + ANSStateT(v) * read);
              // outsym = (outsym << 8) | symbol[s_bar];
              // outBlock_[k + j4] = symbol[s_bar];
              tempoutsym[3] = symbol[s_bar];

              temp --;
              s_bar = state[temp] & StateMask;
              state[temp] = pdf[s_bar] * (state[temp] >> ProbBits) + ANSStateT(cdf[s_bar]);
              read = state[temp] < kANSMinState;
              compressedWords -= read;
              v = blockDataIn[compressedWords];
              state[temp] = ((state[temp] << (kANSEncodedBits * read)) + ANSStateT(v) * read);
              // outsym = (outsym << 8) | symbol[s_bar];
              // outBlock_[k + j5] = symbol[s_bar];
              tempoutsym[2] = symbol[s_bar];

              temp --;
              s_bar = state[temp] & StateMask;
              state[temp] = pdf[s_bar] * (state[temp] >> ProbBits) + ANSStateT(cdf[s_bar]);
              read = state[temp] < kANSMinState;
              compressedWords -= read;
              v = blockDataIn[compressedWords];
              state[temp] = ((state[temp] << (kANSEncodedBits * read)) + ANSStateT(v) * read);
              // outsym = (outsym << 8) | symbol[s_bar];
              // outBlock_[k + j6] = symbol[s_bar];
              tempoutsym[1] = symbol[s_bar];

              temp --;
              s_bar = state[temp] & StateMask;
              state[temp] = pdf[s_bar] * (state[temp] >> ProbBits) + ANSStateT(cdf[s_bar]);
              read = state[temp] < kANSMinState;
              compressedWords -= read;
              v = blockDataIn[compressedWords];
              state[temp] = ((state[temp] << (kANSEncodedBits * read)) + ANSStateT(v) * read);
              // outsym = (outsym << 8) | symbol[s_bar];
              // outBlock_[k + j7] = symbol[s_bar];
              tempoutsym[0] = symbol[s_bar];

              ((uint64_t*)outBlock_)[tempk + l] = ((uint64_t*)tempoutsym)[0];
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
                compressedWords -= read;
                auto v = blockDataIn[compressedWords];
                state[j] = ((state[j] << (kANSEncodedBits * read)) + ANSStateT(v) * read);
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
                compressedWords -= read;
                auto v = blockDataIn[compressedWords];
                state[j] = ((state[j] << (kANSEncodedBits * read)) + ANSStateT(v) * read);
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