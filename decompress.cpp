#include "CpuANSDecode.h"

using namespace cpu_ans;

void decompressFileWithANS(
		const std::string& tempFilePath,
		const std::string& outputFilePath, 
		int precision) {
    std::ifstream inFile0(tempFilePath, std::ios::binary);
    std::vector<uint8_t> fileCompressedHead(32);
    inFile0.read(reinterpret_cast<char*>(fileCompressedHead.data()), 32);
    auto Header = (ANSCoalescedHeader*)fileCompressedHead.data();
    int totalCompressedSize = Header->getTotalCompressedSize();
    int batchSize = Header->getTotalUncompressedWords();
    inFile0.close();
    std::ifstream inFile1(tempFilePath, std::ios::binary);
    std::vector<uint8_t> fileCompressedData(totalCompressedSize);
    inFile1.read(reinterpret_cast<char*>(fileCompressedData.data()), totalCompressedSize);
    inFile1.close();
    uint8_t* decPtrs = (uint8_t*)malloc(sizeof(uint8_t)*(batchSize));
    uint32_t* symbol = (uint32_t*)std::aligned_alloc(kBlockAlignment, sizeof(uint32_t) * (1 << precision));
    uint32_t* pdf = (uint32_t*)std::aligned_alloc(kBlockAlignment, sizeof(uint32_t) * (1 << precision));
    uint32_t* cdf = (uint32_t*)std::aligned_alloc(kBlockAlignment, sizeof(uint32_t) * (1 << precision));
    std::cout<<"decode start!"<<std::endl;
    double decomp_time = 999999;
    for(int i = 0; i < 11; i ++){
    auto start = std::chrono::high_resolution_clock::now();
    ansDecode(
        symbol,
        pdf,
        cdf,
        precision,
        fileCompressedData.data(),
        decPtrs);
    auto end = std::chrono::high_resolution_clock::now();  
    if(decomp_time > std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1e3)
        decomp_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1e3; 
    }
    double dc_bw = ( 1.0 * totalCompressedSize / 1e6 ) / ( (decomp_time) * 1e-3 );
    std::cout << "decomp time " << std::fixed << std::setprecision(6) << (decomp_time) << " ms B/W "   
                  << std::fixed << std::setprecision(1) << dc_bw << " MB/s" << std::endl;
    std::ofstream outFile(outputFilePath, std::ios::binary);
    outFile.write(reinterpret_cast<const char*>(decPtrs), batchSize*sizeof(uint8_t));
    outFile.close();
    free(decPtrs);
    free(symbol);
    free(pdf);
    free(cdf);
    decPtrs = NULL;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input.file> <output.file>" << std::endl;
        return 1;
    }
    int precision = 10; 
	decompressFileWithANS(
        argv[1],argv[2],
        precision);
    std::cout << "Decompression completed successfully." << std::endl;
    return 0;
}