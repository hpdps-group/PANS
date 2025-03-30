#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <omp.h>
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
    std::cout<<"decode start!"<<std::endl;
    double decomp_time = 0.0;
    for(int i = 0; i < 11; i ++){
    auto start = std::chrono::high_resolution_clock::now();
    ansDecode(
        precision,
        fileCompressedData.data(),
        decPtrs);
    auto end = std::chrono::high_resolution_clock::now();  
    if(i > 5)
        decomp_time += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1e3; 
    }
    double dc_bw = ( 1.0 * totalCompressedSize / 1e9 ) / ( (decomp_time / 5.0) * 1e-3 );
    std::cout << "decomp time " << std::fixed << std::setprecision(3) << (decomp_time / 5.0) << " ms B/W "   
                  << std::fixed << std::setprecision(1) << dc_bw << " GB/s" << std::endl;
    std::ofstream outFile(outputFilePath, std::ios::binary);
    outFile.write(reinterpret_cast<const char*>(decPtrs), batchSize*sizeof(uint8_t));
    outFile.close();
    free(decPtrs);
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