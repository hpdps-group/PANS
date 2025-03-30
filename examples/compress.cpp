#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include "ans/GpuANSEncode.h"
#include "ans/GpuANSCodec.h"

using namespace pans_hip;

void compressFileWithANS(
		const std::string& inputFilePath,//输入数据文件路径
		const std::string& tempFilePath,//压缩后文件保存路径
        uint32_t& batchSize,//原本数据规模
		uint32_t& compressedSize,//压缩后数据大小
		int precision,//ANS的精度
		hipStream_t stream
		) {
    //读取输入文件
    std::ifstream inputFile(inputFilePath, std::ios::binary | std::ios::ate);
    std::streamsize fileSize = inputFile.tellg();
    std::vector<uint8_t> fileData(fileSize);
    inputFile.seekg(0, std::ios::beg);
    inputFile.read(reinterpret_cast<char*>(fileData.data()), fileSize);//全部按照uint8_t读入
    inputFile.close();

    //传输输入文件的数据
    uint8_t* inPtrs;
    hipMalloc(&inPtrs, sizeof(uint8_t)*(fileSize));
    hipMemcpy(inPtrs, fileData.data(), fileSize*sizeof(uint8_t), hipMemcpyHostToDevice);

    //设置batchSize，只有一个batch
    batchSize = fileSize;

    //分配存储压缩后数据大小的GPU空间
    uint32_t* outCompressedSize;
    hipMalloc(&outCompressedSize, sizeof(uint32_t));

    //分配存储压缩后数据的GPU空间
    uint8_t* encPtrs;
    hipMalloc(&encPtrs, static_cast<uint64_t>(getMaxCompressedSize(fileSize)));

    std::cout<<"encode start!"<<std::endl;
    //计时
    double comp_time = 0.0;
    for(int i = 0; i < 11; i ++){
    auto start = std::chrono::high_resolution_clock::now();  

    //压缩开始 
    ansEncode(
        precision,
        inPtrs,//已经在dev
        batchSize,
        encPtrs,//GPU
        outCompressedSize,//GPU
        stream);
    hipStreamSynchronize(stream);

    auto end = std::chrono::high_resolution_clock::now();
    if(i > 5)
        comp_time += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1e3;  
    }
    //计算速度
    double c_bw = ( 1.0 * fileSize / 1e9 ) / ( (comp_time / 5.0) * 1e-3 );  
    //输出结果
    std::cout << "comp   time " << std::fixed << std::setprecision(6) << comp_time / 5.0 << " ms B/W "   
                  << std::fixed << std::setprecision(1) << c_bw << " GB/s " << std::endl;
    
    //获取压缩后的数据大小
    uint32_t outsize;
    hipMemcpy(&outsize, outCompressedSize, sizeof(uint32_t), hipMemcpyDeviceToHost);
    compressedSize = outsize;
    //printf("compressed size %d\n", compressedSize);

    //保存压缩后的数据到tempFilePath
    std::ofstream outputFile(tempFilePath, std::ios::binary);
    std::vector<uint8_t> compressedData(outsize);
    hipMemcpy(compressedData.data(), encPtrs, outsize*sizeof(uint8_t), hipMemcpyDeviceToHost);
    outputFile.write(reinterpret_cast<const char*>(compressedData.data()), outsize*sizeof(uint8_t));
    outputFile.close();
}

// void decompressFileWithANS(
// 		const std::string& tempFilePath, //压缩文件路径
// 		const std::string& outputFilePath,   //解压缩后文件路径
//         uint32_t& batchSize,      //解压缩后的数据大小，原本数据大小      
//         uint32_t& compressedSize, //压缩后的数据大小          
// 		int precision,//精度
// 		hipStream_t stream) {
//     //读取压缩文件
//     std::ifstream inFile(tempFilePath, std::ios::binary);
//     std::vector<uint8_t> fileCompressedData(compressedSize);
//     inFile.read(reinterpret_cast<char*>(fileCompressedData.data()), compressedSize);
//     inFile.close();

//     uint8_t* filePtrs;//传输输入数据
//     hipMalloc(&filePtrs, sizeof(uint8_t)*(compressedSize));
//     hipMemcpy(filePtrs,fileCompressedData.data(),compressedSize*sizeof(uint8_t),hipMemcpyHostToDevice);

//     uint8_t* decPtrs;//分配保存解压缩数据的空间
//     hipMalloc(&decPtrs, sizeof(uint8_t)*(batchSize));
    
//     std::cout<<"decode start!"<<std::endl;
//     //计时
//     double decomp_time = 0.0;
//     auto start = std::chrono::high_resolution_clock::now();

//     //解压开始
//     ansDecode(
//         precision,//解压缩精度
//         filePtrs, //解压缩输入数据
//         decPtrs,//解压缩输出数据
//         stream);
//     hipStreamSynchronize(stream);
//     //printf("1\n");
//     auto end = std::chrono::high_resolution_clock::now();  
//     decomp_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1e3; 
    
//     //计算速度
//     double dc_bw = ( 1.0 * compressedSize / 1e9 ) / ( decomp_time * 1e-3 );
//     //输出结果
//     std::cout << "decomp time " << std::fixed << std::setprecision(3) << decomp_time << " ms B/W "   
//                   << std::fixed << std::setprecision(1) << dc_bw << " GB/s" << std::endl;
//     //保存解压后的文件到outputFilePath
//     std::ofstream outFile(outputFilePath, std::ios::binary);
//     std::vector<uint8_t> unCompressData(batchSize);
//     hipMemcpy(unCompressData.data(),decPtrs,batchSize*sizeof(uint8_t),hipMemcpyDeviceToHost);
//     outFile.write(reinterpret_cast<const char*>(unCompressData.data()), batchSize*sizeof(uint8_t));
//     outFile.close();
// }

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <inputfile> <tempfile> " << std::endl;
        return 1;
    }
    hipStream_t stream;   
    hipStreamCreate(&stream);
    uint32_t batchSize;
    uint32_t compressedSize;
    int precision = 10; 
    compressFileWithANS(
        argv[1], argv[2],
        batchSize,//压缩前数据的大小
        compressedSize,//压缩后的数据大小
        precision,//ANS的精度
        stream);
    printf("compress ratio: %f\n", 1.0 * batchSize / compressedSize);
	// decompressFileWithANS(
    //     argv[3],argv[4],
    //     batchSize,//原本的数据规模s
    //     compressedSize,//压缩后数据规模
    //     precision,//精度
    //     stream);
    std::cout << "Compression completed successfully." << std::endl;
    return 0;
}
