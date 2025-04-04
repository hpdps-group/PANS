
#ifndef CPU_ANS_INCLUDE_ANS_CPUANSDECODE_H
#define CPU_ANS_INCLUDE_ANS_CPUANSDECODE_H

#pragma once

#include "CpuANSUtils.h"

namespace cpu_ans {

struct SymbolInfo {
  uint32_t x;//symbol
  uint32_t y;//pdf
  uint32_t z;//cdf
};

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
    uint32_t* symbol,
    uint32_t* pdf,
    uint32_t* cdf,
    void* in,
    void* out
    ) {
  int num_threads = 16;
  auto headerIn = (ANSCoalescedHeader*)in;
  auto opdf = headerIn->getSymbolProbs();
  // __builtin_prefetch(opdf, 0, 3);
  std::vector<uint32_t> ocdf(kNumSymbols);
  std::exclusive_scan(opdf, opdf + kNumSymbols, ocdf.begin(), 0);
  std::vector<SymbolInfo> symbol_info(1 << ProbBits);
  // uint32_t* symbol = (uint32_t*)std::aligned_alloc(kBlockAlignment, sizeof(uint32_t) * (1 << ProbBits));
  // uint32_t* pdf = (uint32_t*)std::aligned_alloc(kBlockAlignment, sizeof(uint32_t) * (1 << ProbBits));
  // uint32_t* cdf = (uint32_t*)std::aligned_alloc(kBlockAlignment, sizeof(uint32_t) * (1 << ProbBits));
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
        cdf[j] = (uint32_t)k;
        // symbol_info[j] = {i, smempdf, (uint32_t)k};
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
            SymbolInfo info;
            int tempk = k >> 3;
            uint8_t tempoutsym[8];
            // __builtin_prefetch(blockDataIn + compressedWords, 0, 0);
            for(int j = kWarpSize - 8, l = 3; j >= 0; j -= 8, l --){
              int temp = j + 7;
              s_bar = state[temp] & StateMask;
              // info = symbol_info[s_bar];
              // state[temp] = info.y * (state[temp] >> ProbBits) + ANSStateT(info.z);
              // read = state[temp] < kANSMinState;
              // compressedWords -= read;
              // v = blockDataIn[compressedWords];
              // state[temp] = ((state[temp] << (kANSEncodedBits * read)) + ANSStateT(v) * read);
              // tempoutsym[7] = info.x;
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
              // info = symbol_info[s_bar];
              // state[temp] = info.y * (state[temp] >> ProbBits) + ANSStateT(info.z);
              // read = state[temp] < kANSMinState;
              // compressedWords -= read;
              // v = blockDataIn[compressedWords];
              // state[temp] = ((state[temp] << (kANSEncodedBits * read)) + ANSStateT(v) * read);
              // tempoutsym[6] = info.x;
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
              // info = symbol_info[s_bar];
              // state[temp] = info.y * (state[temp] >> ProbBits) + ANSStateT(info.z);
              // read = state[temp] < kANSMinState;
              // compressedWords -= read;
              // v = blockDataIn[compressedWords];
              // state[temp] = ((state[temp] << (kANSEncodedBits * read)) + ANSStateT(v) * read);
              // tempoutsym[5] = info.x;
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
              // info = symbol_info[s_bar];
              // state[temp] = info.y * (state[temp] >> ProbBits) + ANSStateT(info.z);
              // read = state[temp] < kANSMinState;
              // compressedWords -= read;
              // v = blockDataIn[compressedWords];
              // state[temp] = ((state[temp] << (kANSEncodedBits * read)) + ANSStateT(v) * read);
              // tempoutsym[4] = info.x;
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
              // info = symbol_info[s_bar];
              // state[temp] = info.y * (state[temp] >> ProbBits) + ANSStateT(info.z);
              // read = state[temp] < kANSMinState;
              // compressedWords -= read;
              // v = blockDataIn[compressedWords];
              // state[temp] = ((state[temp] << (kANSEncodedBits * read)) + ANSStateT(v) * read);
              // tempoutsym[3] = info.x;
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
              // info = symbol_info[s_bar];
              // state[temp] = info.y * (state[temp] >> ProbBits) + ANSStateT(info.z);
              // read = state[temp] < kANSMinState;
              // compressedWords -= read;
              // v = blockDataIn[compressedWords];
              // state[temp] = ((state[temp] << (kANSEncodedBits * read)) + ANSStateT(v) * read);
              // tempoutsym[2] = info.x;
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
              // info = symbol_info[s_bar];
              // state[temp] = info.y * (state[temp] >> ProbBits) + ANSStateT(info.z);
              // read = state[temp] < kANSMinState;
              // compressedWords -= read;
              // v = blockDataIn[compressedWords];
              // state[temp] = ((state[temp] << (kANSEncodedBits * read)) + ANSStateT(v) * read);
              // tempoutsym[1] = info.x;
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
              // info = symbol_info[s_bar];
              // state[temp] = info.y * (state[temp] >> ProbBits) + ANSStateT(info.z);
              // read = state[temp] < kANSMinState;
              // compressedWords -= read;
              // v = blockDataIn[compressedWords];
              // state[temp] = ((state[temp] << (kANSEncodedBits * read)) + ANSStateT(v) * read);
              // tempoutsym[0] = info.x;
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
              for(int j = remainder - 1; j >= 0; j --){
                  // bool valid = j < remainder;
                  auto s_bar = state[j] & StateMask;
                  // if(valid){
                  state[j] = pdf[s_bar] * (state[j] >> ProbBits) + ANSStateT(cdf[s_bar]);
                  // }
                  bool read = 
                  // valid && 
                  (state[j] < kANSMinState);
                  compressedWords -= read;
                  auto v = blockDataIn[compressedWords];
                  state[j] = ((state[j] << (kANSEncodedBits * read)) + ANSStateT(v) * read);
                  // if(valid){
                  outBlock_[uncompressedOffset + j] = symbol[s_bar];
                  // }
                //   if(valid){
                //       state[j] = pdf[s_bar] * (state[j] >> ProbBits) + ANSStateT(cdf[s_bar]);
                //   }
                //   bool read = valid && (state[j] < kANSMinState);
                // compressedWords -= read;
                // auto v = blockDataIn[compressedWords];
                // state[j] = ((state[j] << (kANSEncodedBits * read)) + ANSStateT(v) * read);
                //   if(valid){
                //       outBlock_[uncompressedOffset + j] = symbol[s_bar];
                //   }
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
    uint32_t* symbol,
    uint32_t* pdf,
    uint32_t* cdf,
    int precision,
    uint8_t* in,
    uint8_t* out
    ) {
  
  {
#define RUN_DECODE(BITS)                                           \
  do { ansDecodeKernel_opti<BITS, kDefaultBlockSize>(symbol, pdf, cdf, in, out);} while (false)   \
    
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

// #ifndef CPU_ANS_INCLUDE_ANS_CPUANSDECODE_H
// #define CPU_ANS_INCLUDE_ANS_CPUANSDECODE_H

// #pragma once

// #include <immintrin.h>
// #include <vector>
// #include <numeric>
// #include <iostream>
// #include "CpuANSUtils.h"

// namespace cpu_ans {

// template <int ProbBits, int kDefaultBlockSize>
// void ansDecodeKernel_opti(void* in, void* out) {
//   constexpr int kVectorsPerWarp = kWarpSize / 8;
//   int num_threads = 16;
//   auto headerIn = (ANSCoalescedHeader*)in;
//   auto opdf = headerIn->getSymbolProbs();
  
//   // Build symbol lookup tables
//   std::vector<uint32_t> ocdf(kNumSymbols);
//   std::exclusive_scan(opdf, opdf + kNumSymbols, ocdf.begin(), 0);
  
//   uint32_t* symbol = static_cast<uint32_t*>(
//       std::aligned_alloc(kBlockAlignment, sizeof(uint32_t) * (1 << ProbBits)));
//   uint32_t* pdf = static_cast<uint32_t*>(
//       std::aligned_alloc(kBlockAlignment, sizeof(uint32_t) * (1 << ProbBits)));
//   uint32_t* cdf = static_cast<uint32_t*>(
//       std::aligned_alloc(kBlockAlignment, sizeof(uint32_t) * (1 << ProbBits)));

//   // Initialize lookup tables
//   for (uint32_t i = 0; i < kNumSymbols; ++i) {
//     auto smempdf = opdf[i];
//     auto begin = ocdf[i];
//     for (uint32_t j = begin, k = 0; j < begin + smempdf; ++j, ++k) {
//       symbol[j] = i;
//       pdf[j] = smempdf;
//       cdf[j] = k;
//     }
//   }

//   auto header = *headerIn;
//   const auto numBlocks = header.getNumBlocks();
//   constexpr ANSStateT StateMask = (ANSStateT(1) << ProbBits) - ANSStateT(1);
//   const auto blockWordspre = headerIn->getBlockWords(numBlocks);

//   #pragma omp parallel proc_bind(spread) num_threads(num_threads)
//   {
//     #pragma omp for schedule(dynamic, 8)
//     for (int i = 0; i < numBlocks; ++i) {
//       ANSStateT state[kWarpSize];
//       const auto State = headerIn->getWarpStates()[i].warpState;
//       for (int j = 0; j < kWarpSize; ++j) {
//         state[j] = State[j];
//       }

//       const auto blockWords = blockWordspre[i];
//       uint32_t uncompressedWords = (blockWords.x >> 16);
//       uint32_t compressedWords = (blockWords.x & 0xffff);
//       const uint32_t blockCompressedWordStart = blockWords.y;
//       const ANSEncodedT* blockDataIn =
//           headerIn->getBlockDataStart(numBlocks) + blockCompressedWordStart;
//       uint8_t* outBlock_ = static_cast<uint8_t*>(out) + (i << 12);

//       if (uncompressedWords == kDefaultBlockSize) {
//         for (int k = kDefaultBlockSize - kWarpSize; k >= 0; k -= kWarpSize) {
//           // Process 8 states at a time with AVX2
//           for (int j = kWarpSize - 1; j >= 0; j -= 8) {
//             // Load states
//             __m256i vec_state = _mm256_loadu_si256(
//                 reinterpret_cast<const __m256i*>(&state[j - 7]));

//             // Calculate s_bar = state & StateMask
//             const __m256i vec_mask = _mm256_set1_epi32(StateMask);
//             const __m256i vec_s_bar = _mm256_and_si256(vec_state, vec_mask);

//             // Gather PDF and CDF values
//             __m256i vec_pdf = _mm256_i32gather_epi32(
//                 reinterpret_cast<const int*>(pdf), vec_s_bar, 4);
//             __m256i vec_cdf = _mm256_i32gather_epi32(
//                 reinterpret_cast<const int*>(cdf), vec_s_bar, 4);

//             // Calculate new_state = (state >> ProbBits) * pdf + cdf
//             const __m256i vec_shifted = _mm256_srli_epi32(vec_state, ProbBits);
//             const __m256i vec_mul = _mm256_mullo_epi32(vec_shifted, vec_pdf);
//             const __m256i vec_new_state = _mm256_add_epi32(vec_mul, vec_cdf);

//             // Store new states
//             _mm256_storeu_si256(reinterpret_cast<__m256i*>(&state[j - 7]), vec_new_state);

//             // Check for renormalization and process scalar
//             for (int m = 8; m > 0; m --) {
//               const int idx = j - 7 + m;
//               if (state[idx] < kANSMinState) {
//                 --compressedWords;
//                 const auto v = blockDataIn[compressedWords];
//                 state[idx] = (state[idx] << kANSEncodedBits) + v;
//               }
//             }

//             // Gather symbols and store
//             __m256i vec_symbol = _mm256_i32gather_epi32(
//                 reinterpret_cast<const int*>(symbol), vec_s_bar, 4);
//             alignas(32) uint32_t symbols[8];
//             _mm256_store_si256(reinterpret_cast<__m256i*>(symbols), vec_symbol);
            
//             #pragma omp simd
//             for (int m = 8; m > 0; m --) {
//               outBlock_[k + (j - 7 + m)] = static_cast<uint8_t>(symbols[m]);
//             }
//           }
//         }
//       } else {
//         // Remainder handling (similar optimization pattern)
//         uint32_t remainder = uncompressedWords & 31;
//         int uncompressedOffset = uncompressedWords - remainder;
//         // ... (similar AVX2 pattern for remainder handling)
//       }
//     }
//   }

//   std::free(symbol);
//   std::free(pdf);
//   std::free(cdf);
// }

// void ansDecode(int precision, uint8_t* in, uint8_t* out) {
//   {
// #define RUN_DECODE(BITS) \
//     do { ansDecodeKernel_opti<BITS, kDefaultBlockSize>(in, out); } while (false)

//     switch (precision) {
//       case 9:  RUN_DECODE(9);  break;
//       case 10: RUN_DECODE(10); break;
//       case 11: RUN_DECODE(11); break;
//       default: std::cout << "Unhandled precision: " << precision << std::endl;
//     }
// #undef RUN_DECODE
//   }
// }

// } // namespace cpu_ans

// #endif