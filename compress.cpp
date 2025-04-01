#include "CpuANSEncode.h"

using namespace cpu_ans;

void compressFileWithANS(
		const std::string& inputFilePath,//输入数据文件路径
		const std::string& tempFilePath,//压缩后文件保存路径
        uint32_t& batchSize,//原本数据规模
		uint32_t& compressedSize,//压缩后数据大小
		int precision
		) {
    std::ifstream inputFile(inputFilePath, std::ios::binary | std::ios::ate);
    std::streamsize fileSize = inputFile.tellg();
    std::vector<uint8_t> fileData(fileSize);
    inputFile.seekg(0, std::ios::beg);
    inputFile.read(reinterpret_cast<char*>(fileData.data()), fileSize);//全部按照uint8_t读入
    inputFile.close();

    uint8_t* inPtrs = fileData.data();

    batchSize = fileSize;

    uint32_t* outCompressedSize = (uint32_t*)malloc(sizeof(uint32_t));
    uint8_t* encPtrs = (uint8_t*)malloc(getMaxCompressedSize(fileSize));
    ANSCoalescedHeader* headerOut = (ANSCoalescedHeader*)encPtrs;
    uint32_t maxNumCompressedBlocks;

    uint32_t maxUncompressedWords = fileSize / sizeof(ANSDecodedT);
    maxNumCompressedBlocks =
        (maxUncompressedWords + kDefaultBlockSize - 1) / kDefaultBlockSize;//一个batch的数据以kDefaultBlockSize作为基准划分数据，形成多个数据块
    
    uint4* table = (uint4*)malloc(sizeof(uint4) * kNumSymbols);
    uint32_t* tempHistogram = (uint32_t*)malloc(sizeof(uint32_t) * kNumSymbols);
    uint32_t uncoalescedBlockStride = getMaxBlockSizeUnCoalesced(kDefaultBlockSize);
    uint8_t* compressedBlocks_host = (uint8_t*)std::aligned_alloc(kBlockAlignment, sizeof(uint8_t) * maxNumCompressedBlocks * uncoalescedBlockStride);
    uint32_t* compressedWords_host = (uint32_t*)std::aligned_alloc(kBlockAlignment, sizeof(uint32_t) * maxNumCompressedBlocks);
    uint32_t* compressedWords_host_prefix = (uint32_t*)std::aligned_alloc(kBlockAlignment, sizeof(uint32_t) * maxNumCompressedBlocks);
    uint32_t* compressedWordsPrefix_host = (uint32_t*)std::aligned_alloc(kBlockAlignment, sizeof(uint32_t) * maxNumCompressedBlocks);
    std::cout<<"encode start!"<<std::endl;
    double comp_time = 0.0;
    for(int i = 0; i < 11; i ++){
    auto start = std::chrono::high_resolution_clock::now();  

    ansEncode(
        table,
        tempHistogram,
        precision,
        inPtrs,
        batchSize,
        encPtrs,
        outCompressedSize,
        headerOut,
        maxNumCompressedBlocks,
        uncoalescedBlockStride,
        compressedBlocks_host,
        compressedWords_host,
        compressedWords_host_prefix,
        compressedWordsPrefix_host);

    auto end = std::chrono::high_resolution_clock::now();
    if(i > 5)
      comp_time += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1e3;  
    }
    double c_bw = ( 1.0 * fileSize / 1e6 ) / ( (comp_time / 5.0) * 1e-3 );  
    std::cout << "comp   time " << std::fixed << std::setprecision(3) << comp_time / 5.0 << " ms B/W "   
                  << std::fixed << std::setprecision(1) << c_bw << " MB/s " << std::endl;

    std::ofstream outputFile(tempFilePath, std::ios::binary);
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
          (kDefaultBlockSize << 16) | compressedWords_host[i], 
          compressedWordsPrefix_host[i]};
    }
    auto uncoalescedBlock = compressedBlocks_host + i * uncoalescedBlockStride;
    for(int j = 0; j < kWarpSize; ++j){
      auto warpStateOut = (ANSWarpState*)uncoalescedBlock;
      headerOut->getWarpStates()[i].warpState[j] = (warpStateOut->warpState[j]);
    }
    
    uint32_t lastBlockWords = fileSize % kDefaultBlockSize;
    lastBlockWords = lastBlockWords == 0 ? kDefaultBlockSize : lastBlockWords;

    blockWordsOut[i] = uint2{
        (lastBlockWords << 16) | compressedWords_host[i], compressedWordsPrefix_host[i]};
    
    outputFile.write(reinterpret_cast<const char*>(encPtrs), headerOut->getCompressedOverhead(maxNumCompressedBlocks));

    i = 0;
    for(; i < maxNumCompressedBlocks - 1; i ++){
    
      auto uncoalescedBlock = compressedBlocks_host + i * uncoalescedBlockStride;
      uint32_t numWords = compressedWords_host[i];

      uint32_t limitEnd = divUp(numWords, kBlockAlignment / sizeof(ANSEncodedT));

      auto inT = (const uint4*)(uncoalescedBlock + sizeof(ANSWarpState));
      auto outT = (uint4*)(BlockDataStart + compressedWordsPrefix_host[i]);

      //   __builtin_memcpy(outT, inT, limitEnd << 4);
      outputFile.write(reinterpret_cast<const char*>(inT), limitEnd << 4);
    }
    // uncoalescedBlock = compressedBlocks_host + i * uncoalescedBlockStride;

    uint32_t numWords = compressedWords_host[i];

    uint32_t limitEnd = divUp(numWords, kBlockAlignment / sizeof(ANSEncodedT));

    auto inT = (const uint4*)(uncoalescedBlock + sizeof(ANSWarpState));
    auto outT = (uint4*)(BlockDataStart + compressedWordsPrefix_host[i]);

    // __builtin_memcpy,(outT, inT, limitEnd << 4);
    outputFile.write(reinterpret_cast<const char*>(inT), limitEnd << 4);

    uint32_t outsize = *outCompressedSize;
    compressedSize = outsize;

    // outputFile.write(reinterpret_cast<const char*>(encPtrs), outsize*sizeof(uint8_t));
    outputFile.close();
    free(outCompressedSize);
    free(encPtrs);
    free(table);
    free(tempHistogram);
    free(compressedBlocks_host);
    free(compressedWords_host);
    free(compressedWords_host_prefix);
    free(compressedWordsPrefix_host);
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input.file> <output.file>" << std::endl;
        return 1;
    }
    uint32_t batchSize;
    uint32_t compressedSize;
    int precision = 10; 
    compressFileWithANS(
        argv[1], argv[2],
        batchSize,
        compressedSize,
        precision);
        // printf("compressedSize: %d\n", compressedSize);
    printf("compress ratio: %f\n", 1.0 * batchSize / compressedSize);
    std::cout << "Compression completed successfully." << std::endl;
    return 0;
}