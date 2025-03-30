#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include "ans/GpuANSDecode.h"
#include "ans/GpuANSCodec.h"

using namespace pans_hip;

void decompressFileWithANS(
		const std::string& tempFilePath, //压缩文件路径
		const std::string& outputFilePath,   //解压缩后文件路径
        uint32_t& batchSize,      //解压缩后的数据大小，原本数据大小      
        uint32_t& compressedSize, //压缩后的数据大小          
		int precision,//精度
		hipStream_t stream) {
    std::ifstream inFile0(tempFilePath, std::ios::binary);
    std::vector<uint8_t> fileCompressedHead(32);
    inFile0.read(reinterpret_cast<char*>(fileCompressedHead.data()), 32);
    auto Header = (ANSCoalescedHeader*)fileCompressedHead.data();
    compressedSize = Header->getTotalCompressedSize();
    batchSize = Header->getTotalUncompressedWords();
    // printf("batchSize: %d\n", batchSize);
    inFile0.close();
    // printf("totalCompressedSize: %d\n", totalCompressedSize);
    std::ifstream inFile1(tempFilePath, std::ios::binary);
    std::vector<uint8_t> fileCompressedData(compressedSize);
    inFile1.read(reinterpret_cast<char*>(fileCompressedData.data()), compressedSize);
    inFile1.close();
    uint8_t* filePtrs;
    hipMalloc(&filePtrs, sizeof(uint8_t)*(compressedSize));
    hipMemcpy(filePtrs,fileCompressedData.data(),compressedSize*sizeof(uint8_t),hipMemcpyHostToDevice);

    uint8_t* decPtrs;
    hipMalloc(&decPtrs, sizeof(uint8_t)*(batchSize));
    
    std::cout<<"decode start!"<<std::endl;
    double decomp_time = 0.0;
    auto start = std::chrono::high_resolution_clock::now();

    //解压开始
    ansDecode(
        precision,//解压缩精度
        filePtrs, //解压缩输入数据
        decPtrs,//解压缩输出数据
        stream);
    hipStreamSynchronize(stream);
    //printf("1\n");
    auto end = std::chrono::high_resolution_clock::now();  
    decomp_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1e3; 
    
    //计算速度
    double dc_bw = ( 1.0 * compressedSize / 1e9 ) / ( decomp_time * 1e-3 );
    //输出结果
    std::cout << "decomp time " << std::fixed << std::setprecision(3) << decomp_time << " ms B/W "   
                  << std::fixed << std::setprecision(1) << dc_bw << " GB/s" << std::endl;
    //保存解压后的文件到outputFilePath
    std::ofstream outFile(outputFilePath, std::ios::binary);
    std::vector<uint8_t> unCompressData(batchSize);
    hipMemcpy(unCompressData.data(),decPtrs,batchSize*sizeof(uint8_t),hipMemcpyDeviceToHost);
    outFile.write(reinterpret_cast<const char*>(unCompressData.data()), batchSize*sizeof(uint8_t));
    outFile.close();
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input.file> <output.file>" << std::endl;
        return 1;
    }
    hipStream_t stream;   
    hipStreamCreate(&stream);
    uint32_t batchSize;
    uint32_t compressedSize;
    int precision = 10; 
	decompressFileWithANS(
        argv[1],argv[2],
        batchSize,//原本的数据规模
        compressedSize,//压缩后数据规模
        precision,//精度
        stream);
    std::cout << "Decompression completed successfully." << std::endl;
    return 0;
}
