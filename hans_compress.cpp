
#include <fstream>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <cassert>
#include <vector>
#include <cstring>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <numeric> 
#include <thread>
#include <parallel/algorithm>
#include <cstdlib>
#include <stdexcept>
#include <chrono>
#include <atomic>
#include <omp.h>
#include <bitset>

int offset[8][32][2] = {
    //max_bit_length = 1
    {
        {0,0}, {0,1}, {0,2}, {0,3}, {0,4}, {0,5}, {0,6}, {0,7}, //0-7
        {1,0}, {1,1}, {1,2}, {1,3}, {1,4}, {1,5}, {1,6}, {1,7}, //8-15
        {2,0}, {2,1}, {2,2}, {2,3}, {2,4}, {2,5}, {2,6}, {2,7}, //16-23
        {2,8}, {2,9}, {2,10}, {2,11}, {2,12}, {2,13}, {2,14}, {2,15}, //24-31
    },
    //max_bit_length = 2
    {
        {0,0}, {0,2}, {0,4}, {0,6}, {1,0}, {1,2}, {1,4}, {1,6}, //0-7
        {2,0}, {2,2}, {2,4}, {2,6}, {3,0}, {3,2}, {3,4}, {3,6}, //8-15
        {4,0}, {4,2}, {4,4}, {4,6}, {5,0}, {5,2}, {5,4}, {5,6}, //16-23
        {6,0}, {6,2}, {6,4}, {6,6}, {6,8}, {6,10}, {6,12}, {6,14}, //24-31
    },
    //max_bit_length = 3
    {
        {0,0}, {0,3}, {0,6}, {1,1}, {1,4}, {1,7}, {2,2}, {2,5}, //0-7
        {3,0}, {3,3}, {3,6}, {4,1}, {4,4}, {4,7}, {5,2}, {5,5}, //8-15
        {6,0}, {6,3}, {6,6}, {7,1}, {7,4}, {7,7}, {8,2}, {8,5}, //16-23
        {9,0}, {9,3}, {9,6}, {10,1}, {10,4}, {10,7}, {10,10}, {10,13}, //24-31
    },
    //max_bit_length = 4
    {
        {0,0}, {0,4}, {1,0}, {1,4}, {2,0}, {2,4}, {3,0}, {3,4}, //0-7
        {4,0}, {4,4}, {5,0}, {5,4}, {6,0}, {6,4}, {7,0}, {7,4}, //8-15
        {8,0}, {8,4}, {9,0}, {9,4}, {10,0}, {10,4}, {11,0}, {11,4}, //16-23
        {12,0}, {12,4}, {13,0}, {13,4}, {14,0}, {14,4}, {14,8}, {14,12}, //24-31
    },
    //max_bit_length = 5
    {
        {0,0}, {0,5}, {1,2}, {1,7}, {2,4}, {3,1}, {3,6}, {4,3}, //0-7
        {5,0}, {5,5}, {6,2}, {6,7}, {7,4}, {8,1}, {8,6}, {9,3}, //8-15
        {10,0}, {10,5}, {11,2}, {11,7}, {12,4}, {13,1}, {13,6}, {14,3}, //16-23
        {15,0}, {15,5}, {16,2}, {16,7}, {17,4}, {18,1}, {18,6}, {18,11}, //24-31
    },
    //max_bit_length = 6
    {
        {0,0}, {0,6}, {1,4}, {2,2}, {3,0}, {3,6}, {4,4}, {5,2}, //0-7
        {6,0}, {6,6}, {7,4}, {8,2}, {9,0}, {9,6}, {10,4}, {11,2}, //8-15
        {12,0}, {12,6}, {13,4}, {14,2}, {15,0}, {15,6}, {16,4}, {17,2}, //16-23
        {18,0}, {18,6}, {19,4}, {20,2}, {21,0}, {21,6}, {22,4}, {22,10}, //24-31
    },
    //max_bit_length = 7
    {
        {0,0}, {0,7}, {1,6}, {2,5}, {3,4}, {4,3}, {5,2}, {6,1}, //0-7
        {7,0}, {7,7}, {8,6}, {9,5}, {10,4}, {11,3}, {12,2}, {13,1}, //8-15
        {14,0}, {14,7}, {15,6}, {16,5}, {17,4}, {18,3}, {19,2}, {20,1}, //16-23
        {21,0}, {21,7}, {22,6}, {23,5}, {24,4}, {25,3}, {26,2}, {26,9}, //24-31
    },
    //max_bit_length = 8
    {
        {0,0}, {1,0}, {2,0}, {3,0}, {4,0}, {5,0}, {6,0}, {7,0}, //0-7
        {8,0}, {9,0}, {10,0}, {11,0}, {12,0}, {13,0}, {14,0}, {15,0}, //8-15
        {16,0}, {17,0}, {18,0}, {19,0}, {20,0}, {21,0}, {22,0}, {23,0}, //16-23
        {24,0}, {25,0}, {26,0}, {27,0}, {28,0}, {29,0}, {30,0}, {30,8}, //24-31
    }
};

inline uint32_t getAlignmentRoundUp(size_t alignment, const void* ptr) {
    return (alignment - (reinterpret_cast<uintptr_t>(ptr) % alignment)) % alignment;
}

constexpr uint32_t kAlign = 32;
constexpr uint32_t kNumSymbols = 256;
constexpr uint32_t kDefaultBlockSize = 32;
constexpr uint32_t kBlockAlignment = 16;

struct uint2 { uint32_t x, y; };
struct uint4 { uint32_t x, y, z, w; };

template <typename U, typename V>
inline auto divDown(U a, V b) -> decltype(a + b) {
  return (a / b);
}

template <typename U, typename V>
inline auto divUp(U a, V b) -> decltype(a + b) {
  return (a + b - 1) / b;
}

template <typename U, typename V>
inline auto roundDown(U a, V b) -> decltype(a + b) {
  return divDown(a, b) * b;
}

template <typename U, typename V>
inline auto roundUp(U a, V b) -> decltype(a + b) {
  return divUp(a, b) * b;
}
struct CoalescedHeader {
    static inline uint32_t getCompressedOverhead(uint32_t numBlocks) { //bits
        int kAlignment = kBlockAlignment / sizeof(uint2);
        if (kAlignment == 0) kAlignment = 1;

        return sizeof(CoalescedHeader) + //header
               sizeof(uint16_t) * kNumSymbols + // table
               divUp(3 * numBlocks, 8) + // max_bits_length
               sizeof(uint32_t) * roundUp(numBlocks, kAlignment); // compressed_size and compress_prefix
    }
    
    inline uint32_t getTotalCompressedSize() {
        return getCompressedOverhead()  +
               getTotalCompressedWords();
    }

    inline uint32_t getCompressedOverhead() {
        return getCompressedOverhead(getNumBlocks());
    }

    inline float getCompressionRatio() {
        return static_cast<float>(getTotalCompressedWords()) /
               (static_cast<float>(getTotalUncompressedWords()));
    }

    inline uint32_t getNumBlocks() { return numBlocks; }
    inline void setNumBlocks(uint32_t nb) { numBlocks = nb; }

    inline uint32_t getTotalUncompressedWords() { return totalUncompressedWords; }
    inline void setTotalUncompressedWords(uint32_t words) { totalUncompressedWords = words; }

    inline uint32_t getTotalCompressedWords() { return totalCompressedWords; }
    inline void setTotalCompressedWords(uint32_t words) { totalCompressedWords = words; }

    inline uint16_t* getSymbolTable() { return reinterpret_cast<uint16_t*>(this + 1); }

    inline uint8_t* getMaxbit_length() { return reinterpret_cast<uint8_t*>(getSymbolTable() + kNumSymbols); }

    inline uint2* getBlockWords(uint32_t numBlocks) {
        return reinterpret_cast<uint2*>(getMaxbit_length() + divUp(3 * numBlocks, 8));
    }

    inline uint8_t* getBlockDataStart(uint32_t numBlocks) {
        constexpr int kAlignment = kBlockAlignment / sizeof(uint2) == 0
        ? 1 : kBlockAlignment / sizeof(uint2);
        return reinterpret_cast<uint8_t*>(getBlockWords(numBlocks) + roundUp(numBlocks, kAlignment));
    }

    uint32_t numBlocks;
    uint32_t totalUncompressedWords;
    uint32_t totalCompressedWords;
    uint32_t options;// CPU , NV_GPU, AMD_GPU, NPU
    uint32_t unuse0;
    uint32_t unuse1;
    uint32_t unuse2;
    uint32_t unuse3;
};

inline uint32_t
getRawCompBlockMaxSize(uint32_t uncompressedBlockBytes) {
  return roundUp(
      uncompressedBlockBytes, kBlockAlignment);
}

inline uint32_t getMaxBlockSizeCoalesced(uint32_t uncompressedBlockBytes) {
  return getRawCompBlockMaxSize(uncompressedBlockBytes);
}

inline uint32_t getMaxCompressedSize(uint32_t uncompressedBytes) {
  uint32_t blocks = divUp(uncompressedBytes, kDefaultBlockSize);
  size_t rawSize = CoalescedHeader::getCompressedOverhead(blocks);
  rawSize += (size_t)getMaxBlockSizeCoalesced(kDefaultBlockSize) * blocks;
  rawSize = roundUp(rawSize, sizeof(uint4));
  return rawSize;
}

uint32_t getAlignmentRoundUp(uint32_t alignment, const void* ptr) {
    uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
    uint32_t mod = addr % alignment;
    return mod == 0 ? 0 : alignment - mod;
}

void processBlock(const uint8_t* in, uint32_t size, uint32_t* localHist) {
    if (size > kAlign) {
        __builtin_prefetch(in + kAlign, 0, 0);
    }
    // printf("0221\n");
    uint32_t roundUp = std::min(size, static_cast<uint32_t>(getAlignmentRoundUp(kAlign, in)));
    for (uint32_t i = 0; i < roundUp; ++i) {
        ++localHist[in[i]];
    }

    const uint8_t* alignedIn = in + roundUp;
    uint32_t remaining = size - roundUp;
    uint32_t numChunks = remaining / kAlign;
    // printf("0222\n");
    for (uint32_t i = 0; i < numChunks; ++i) {
        const uint8_t* chunk = alignedIn + i * kAlign;
        if (i + 1 < numChunks) {
            __builtin_prefetch(chunk + kAlign, 0, 0);
        }
        ++localHist[chunk[0]]; ++localHist[chunk[1]]; ++localHist[chunk[2]]; ++localHist[chunk[3]];
        ++localHist[chunk[4]]; ++localHist[chunk[5]]; ++localHist[chunk[6]]; ++localHist[chunk[7]];
        ++localHist[chunk[8]]; ++localHist[chunk[9]]; ++localHist[chunk[10]]; ++localHist[chunk[11]];
        ++localHist[chunk[12]]; ++localHist[chunk[13]]; ++localHist[chunk[14]]; ++localHist[chunk[15]];
        ++localHist[chunk[16]]; ++localHist[chunk[17]]; ++localHist[chunk[18]]; ++localHist[chunk[19]];
        ++localHist[chunk[20]]; ++localHist[chunk[21]]; ++localHist[chunk[22]]; ++localHist[chunk[23]];
        ++localHist[chunk[24]]; ++localHist[chunk[25]]; ++localHist[chunk[26]]; ++localHist[chunk[27]];
        ++localHist[chunk[28]]; ++localHist[chunk[29]]; ++localHist[chunk[30]]; ++localHist[chunk[31]];
    }
    const uint8_t* tail = alignedIn + numChunks * kAlign;
    uint32_t remainingTail = remaining % kAlign;
    
    while(remainingTail >= 8) {
        ++localHist[tail[0]]; ++localHist[tail[1]];
        ++localHist[tail[2]]; ++localHist[tail[3]];
        ++localHist[tail[4]]; ++localHist[tail[5]];
        ++localHist[tail[6]]; ++localHist[tail[7]];
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
    const uint8_t* in,
    uint32_t size,
    uint32_t* out,
    bool multithread = true) {
    std::memset(out, 0, kNumSymbols * sizeof(uint32_t));

    // printf("01\n");
    // for(int i = 0 ; i < size ; i ++) {
    //     out[in[i]]++;
    // }
    if (size < 45 * 100000 || !multithread) {
        alignas(64) uint32_t localHist[kNumSymbols] = {0};
        processBlock(in, size, localHist);
        memcpy(out, localHist, kNumSymbols * sizeof(uint32_t));
        return;
    }

    int numThreads = std::thread::hardware_concurrency();
    // std::cout << "numThreads: " << numThreads << std::endl;
    std::vector<std::thread> threads;
    alignas(64) std::vector<uint32_t> histograms(numThreads * kNumSymbols, 0);
    // printf("02\n");
    const uint32_t blockSize = (size + numThreads * 4 - 1) / (numThreads * 4);
    std::atomic<uint32_t> currentBlock(0);
    for (unsigned t = 0; t < numThreads; ++t) {
        // printf("t: %d\n", t);
        threads.emplace_back([&, t]() {
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            CPU_SET(t % numThreads, &cpuset);
            pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
            // printf("021\n");
            uint32_t* localHist = &histograms[t * kNumSymbols];
            while (true) {
                uint32_t blockIdx = currentBlock.fetch_add(1);
                uint32_t start = blockIdx * blockSize;
                if (start >= size) break;
                uint32_t end = std::min(start + blockSize, size);
                // printf("022\n");
                processBlock(in + start, end - start, localHist);
                // printf("023\n");
            }
        });
    }
    // printf("03\n");
    for (auto& thread : threads) {
        thread.join();
    }
    
    for (unsigned t = 0; t < numThreads; ++t) {
        const uint32_t* src = &histograms[t * kNumSymbols];
        for (int i = 0; i < kNumSymbols; ++i) {
            out[i] += src[i];
        }
    }
}

void ansCalcWeights(
    int probBits,
    uint32_t totalNum,
    const uint32_t* counts,
    uint16_t* table) {
    if (totalNum == 0) return;
    std::vector<uint64_t> sortedPairs(kNumSymbols);
    for (int i = 0; i < kNumSymbols; ++i) {
        sortedPairs[i] = ((uint64_t)counts[i] << 32) | i;
    }

    std::sort(
      sortedPairs.begin(),
      sortedPairs.end(),
      [](uint64_t a, uint64_t b) { return a > b; }
    );

    for (int i = 0; i < kNumSymbols; ++i) {
        table[sortedPairs[i] & 0xFFFFFFFFU] = (uint16_t)i;
    }
    for(int i = 0; i < kNumSymbols; ++i) {
        printf("table[%d] = %d\n", i, table[i]);
    }
}

void ansEncodeBatch(
    uint8_t* in,
    int inSize,
    uint32_t maxNumCompressedBlocks,
    uint32_t uncoalescedBlockStride,
    uint8_t* compressedBlocks_dev,
    uint32_t* compressedWords_host,
    uint32_t* compressedWordsPrefix_host,
    uint32_t* max_length,
    const uint16_t* table) {
    // #pragma omp parallel for proc_bind(spread) num_threads(32)
    int num0 = 0;
    int num1 = 0; 
    printf("maxNumCompressedBlocks: %d\n", maxNumCompressedBlocks);
    for(int l = 0; l < maxNumCompressedBlocks; l ++){
    uint32_t start = l * kDefaultBlockSize;
    auto blockSize =  std::min(start + kDefaultBlockSize, (uint32_t)inSize) - start;
    auto inBlock = in + start;
    auto buffer = (compressedBlocks_dev + l * uncoalescedBlockStride);

    uint16_t temp[32];
    int max_bit_length = 32;
    for(int i = 0; i < blockSize; i ++){
        temp[i] = table[inBlock[i]];
        // if(l == 11010011){
        //     printf("inBlock[%d] = %d\n", i, inBlock[i]);
        //     printf("temp[%d] = %d\n", i, temp[i]);
        //     printf("max_bit_length: %d\n", 32 - max_bit_length);
        // }
        max_bit_length = std::min(max_bit_length, __builtin_clz(temp[i]));
    }
    max_bit_length = max_bit_length == 32 ? 1 : 32 - max_bit_length;
    if(max_bit_length != 8)
    num1++;
    else 
    num0++;
    // max_bit_length -= 1;
    // if(l == 0){
        // printf("%d ", max_bit_length);
    // }
    int array_length = max_bit_length << 2;
    for(int i = 0; i < blockSize; i ++){
        uint16_t* buffer16 = (uint16_t*)(buffer + offset[max_bit_length - 1][i][0]);
        *buffer16 &= (temp[i] << offset[max_bit_length - 1][i][1]);
    }
    max_length[l] = max_bit_length;
    compressedWords_host[l] = array_length;
    // compressedWordsPrefix_host[l] = roundUp(array_length, 8);
    }
    printf("num0: %d\n", num0);
    printf("num1: %d\n", num1);
}

void ansEncodeCoalesceBatch(
    const uint8_t* __restrict__ compressedBlocks_host,
    int uncompressedWords,
    uint32_t maxNumCompressedBlocks,
    uint32_t uncoalescedBlockStride,
    const uint32_t* __restrict__ compressedWords_host,
    uint32_t* __restrict__ compressedWordsPrefix_host,
    uint16_t* __restrict__ table,
    uint8_t* out,
    uint32_t* outSize) {

  CoalescedHeader* headerOut = (CoalescedHeader*)out;
  uint32_t totalCompressedWords = 0;
  if(maxNumCompressedBlocks > 0){
    totalCompressedWords =
        compressedWordsPrefix_host[maxNumCompressedBlocks - 1] +
            roundUp(compressedWords_host[maxNumCompressedBlocks - 1], 8);
  }
    
  CoalescedHeader header;
  header.setNumBlocks(maxNumCompressedBlocks);
  header.setTotalUncompressedWords(uncompressedWords);
  header.setTotalCompressedWords(totalCompressedWords);

  *outSize = header.getTotalCompressedSize();
  *headerOut = header; 

  auto table_res = headerOut->getSymbolTable();
  for (int j = 0; j < kNumSymbols; j ++) {
    table_res[j] = table[j];
  }

  auto blockWordsOut = headerOut->getBlockWords(maxNumCompressedBlocks);
  auto BlockDataStart = headerOut->getBlockDataStart(maxNumCompressedBlocks);

  int i = 0;
  for(; i < maxNumCompressedBlocks - 1; i ++){
    
    auto uncoalescedBlock = compressedBlocks_host + i * uncoalescedBlockStride;

    blockWordsOut[i] = uint2{
        (kDefaultBlockSize << 16) | compressedWords_host[i], compressedWordsPrefix_host[i]};

    uint32_t numWords = compressedWords_host[i];

    uint32_t limitEnd = divUp(numWords, 8);

    auto inT = (const uint4*)(uncoalescedBlock);
    auto outT = (uint4*)(BlockDataStart + compressedWordsPrefix_host[i]);

    memcpy(outT, inT, limitEnd << 4);
  }
  auto uncoalescedBlock = compressedBlocks_host + i * uncoalescedBlockStride;

  uint32_t lastBlockWords = uncompressedWords % kDefaultBlockSize;
  lastBlockWords = lastBlockWords == 0 ? kDefaultBlockSize : lastBlockWords;

  blockWordsOut[i] = uint2{
      (lastBlockWords << 16) | compressedWords_host[i], compressedWordsPrefix_host[i]};

  uint32_t numWords = compressedWords_host[i];

  uint32_t limitEnd = divUp(numWords, 8);

  auto inT = (const uint4*)(uncoalescedBlock);
  auto outT = (uint4*)(BlockDataStart + compressedWordsPrefix_host[i]);

  memcpy(outT, inT, limitEnd << 4);
}

void ansEncode(
    int precision,
    uint8_t* in,
    uint32_t inSize,
    uint8_t* out,
    uint32_t* outSize) {

  uint32_t maxUncompressedbyte = inSize;
  uint32_t maxNumCompressedBlocks =
      (maxUncompressedbyte + kDefaultBlockSize - 1) / kDefaultBlockSize;//每次处理32个byte
  uint16_t* table = (uint16_t*)malloc(sizeof(uint16_t) * kNumSymbols);
  uint32_t* tempHistogram = (uint32_t*)malloc(sizeof(uint32_t) * kNumSymbols);
//   printf("0\n");
  ansHistogram(
      in,
      inSize,
      tempHistogram);

//   printf("1\n");
  ansCalcWeights(
      precision,
      inSize,
      tempHistogram,
      table);
//   printf("2\n");

  uint32_t* max_length = (uint32_t*)std::malloc(sizeof(uint32_t) * maxNumCompressedBlocks); 
  uint32_t uncoalescedBlockStride = kDefaultBlockSize;
  uint8_t* compressedBlocks_host = (uint8_t*)std::malloc(sizeof(uint8_t) * maxNumCompressedBlocks * uncoalescedBlockStride);
  uint32_t* compressedWords_host = (uint32_t*)std::malloc(sizeof(uint32_t) * maxNumCompressedBlocks);
  uint32_t* compressedWordsPrefix_host = (uint32_t*)std::malloc(sizeof(uint32_t) * maxNumCompressedBlocks);

  ansEncodeBatch( 
        in,
        inSize,
        maxNumCompressedBlocks,// 压缩块数量
        uncoalescedBlockStride,// 缓冲区
        compressedBlocks_host,// 存储码字
        compressedWords_host,
        compressedWordsPrefix_host,
        max_length,// 存储每个块的截断bit_length
        table);
//   printf("3\n");

  if (maxNumCompressedBlocks > 0)
    std::exclusive_scan(compressedWords_host, compressedWords_host + maxNumCompressedBlocks, compressedWordsPrefix_host, 0);
    printf("compressedWordsPrefix_host: %d\n", compressedWordsPrefix_host[maxNumCompressedBlocks - 1]);
    printf("compressedWords_host: %d\n", compressedWords_host[maxNumCompressedBlocks - 1]);
//   printf("4\n");
  ansEncodeCoalesceBatch(
          compressedBlocks_host,
          inSize,
          maxNumCompressedBlocks,
          uncoalescedBlockStride,
          compressedWords_host,
          compressedWordsPrefix_host,
          table,
          out,
          outSize);
}

void compressFileWithANS(
		const std::string& inputFilePath,
		const std::string& tempFilePath,
        uint32_t& originalSize,
		uint32_t& compressedSize,
		int precision
		) {
    std::ifstream inputFile(inputFilePath, std::ios::binary | std::ios::ate);
    std::streamsize fileSize = inputFile.tellg();
    std::vector<uint8_t> fileData(fileSize);
    inputFile.seekg(0, std::ios::beg);
    inputFile.read(reinterpret_cast<char*>(fileData.data()), fileSize);
    inputFile.close();

    uint8_t* inPtrs = fileData.data();

    uint32_t* outCompressedSize = (uint32_t*)malloc(sizeof(uint32_t));
    uint8_t* encPtrs = (uint8_t*)malloc(getMaxCompressedSize(fileSize));

    std::cout<<"encode start!"<<std::endl;
    auto start = std::chrono::high_resolution_clock::now();  

    ansEncode(
        precision,
        inPtrs,
        fileSize,
        encPtrs,
        outCompressedSize);

    auto end = std::chrono::high_resolution_clock::now();
    double comp_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1e3;  
    double c_bw = ( 1.0 * fileSize / 1e9 ) / ( comp_time * 1e-3 );  
    std::cout << "comp   time " << std::fixed << std::setprecision(3) << comp_time << " ms B/W "   
                  << std::fixed << std::setprecision(1) << c_bw << " GB/s " << std::endl;
    
    uint32_t outsize = *outCompressedSize;
    compressedSize = outsize;
    originalSize = fileSize;
    std::ofstream outputFile(tempFilePath, std::ios::binary);
    outputFile.write(reinterpret_cast<const char*>(encPtrs), outsize*sizeof(uint8_t));
    outputFile.close();
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input.file> <output.file>" << std::endl;
        return 1;
    }
    uint32_t original_Size;
    uint32_t compressedSize;
    int precision = 10; 
    compressFileWithANS(
        argv[1], argv[2],
        original_Size,
        compressedSize,
        precision);
    printf("original size: %d\n", original_Size);
    printf("compressed size: %d\n", compressedSize);
    printf("compress ratio: %f\n", 1.0 * original_Size / compressedSize);
    return 0;
}