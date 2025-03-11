#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <omp.h>
#include "CpuANSEncode.h"

using namespace cpu_ans;

void compressFileWithANS(
		const std::string& inputFilePath,
		const std::string& tempFilePath,
        uint32_t& batchSize,
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

    batchSize = fileSize;

    uint32_t* outCompressedSize = (uint32_t*)malloc(sizeof(uint32_t));
    uint8_t* encPtrs = (uint8_t*)malloc(getMaxCompressedSize(fileSize));

    std::cout<<"encode start!"<<std::endl;
    auto start = std::chrono::high_resolution_clock::now();  

    ansEncode(
        precision,
        inPtrs,
        batchSize,
        encPtrs,
        outCompressedSize);

    auto end = std::chrono::high_resolution_clock::now();
    double comp_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1e3;  
    double c_bw = ( 1.0 * fileSize / 1e9 ) / ( comp_time * 1e-3 );  
    std::cout << "comp   time " << std::fixed << std::setprecision(3) << comp_time << " ms B/W "   
                  << std::fixed << std::setprecision(1) << c_bw << " GB/s " << std::endl;
    
    uint32_t outsize = *outCompressedSize;
    compressedSize = outsize;

    std::ofstream outputFile(tempFilePath, std::ios::binary);
    outputFile.write(reinterpret_cast<const char*>(encPtrs), outsize*sizeof(uint8_t));
    outputFile.close();
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
    printf("compress ratio: %f\n", 1.0 * batchSize / compressedSize);
    return 0;
}