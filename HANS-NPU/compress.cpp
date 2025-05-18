/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
 * This file constains code of cpu debug and npu code.We read data from bin file
 * and write result to file.
 */
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cassert>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include <chrono>

#include "acl/acl.h"
extern "C" __global__ __aicore__ void compress(GM_ADDR x, GM_ADDR y);

// 一个block每次处理32次32个uint32_t(2个uint16_t组成一个uint32_t)，即32 * 32 * 4 = 4096字节数据，每个4096字节数据块产生的max_bits_length数组大小为32字节

int getMaxtempbufferSize(uint32_t byteSize){
    // int MaxtempBufferSize = 0;
    // int bfp16_num = byteSize / 2;
    // if(bfp16_num % 2 == 0) {
    //     int uint32_num = bfp16_num / 2;
    //     int uint32_remainder = bfp16_num % 2;
    // }
    // else {
    //     int uint32_num = bfp16_num / 2;
    //     int uint32_remainder = bfp16_num % 2;
    // }
    // if(uint32_num % 2 == 0) {
    //     int merge_ms_num = uint32_num / 2;
    //     int merge_ms_remainder = uint32_num % 2;
    // }
    // else {
    //     int merge_ms_num = uint32_num / 2;
    //     int merge_ms_remainder = uint32_num % 2;
    // }
    // int align8_num = (byteSize / 8) * 8;// 以字节为单位
    // int align8_remainder = byteSize - align8_num;// remainder位于0-7字节内

    int blockNum = (byteSize + 4096 - 1) / 4096;// 向上取整，最后一个数据块不足4096仍然当作一个块处理
    int block_remainder = byteSize - (byteSize / 4096) * 4096;// 最后一个数据块的数量，以字节为单位
    int MaxtempBufferSize = 2048 * blockNum// 以字节为单位
    // 分别是16字节数据头 + 256个uint32_t的直方图数组 + blockNum * 每个block生成的max_bit_length数组的大小（64 * 0.5字节）+ blockNum * 每个block提取出来的尾数+sign数组 + blockNum * 每个block压缩的指数数组（假设完全没压缩）

    return MaxtempBufferSize;
}

int getFinalbufferSize(uint32_t byteSize){
    // int FinalBufferSize = 0;
    int blockNum = (byteSize + 4096 - 1) / 4096;
    int block_remainder = byteSize - (byteSize / 4096) * 4096;
    int FinalBufferSize = 16 + HISTOGRAM_BINS * sizeof(uint8_t) + 32 * blockNum + 2048 * blockNum + 2048 * blockNum;
    return FinalbufferSize;
}

int main(int32_t argc, char* argv[])
{
    // 获取输入参数
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input.file> <output.file>" << std::endl;
        return 1;
    }

    // 获取输入数据
    struct stat sBuf;
    int fileStatus = stat(argv[1].data(), &sBuf);
    if (fileStatus == -1) {
        ERROR_LOG("failed to get file");
        return false;
    }
    if (S_ISREG(sBuf.st_mode) == 0) {
        ERROR_LOG("%s is not a file, please enter a file", argv[1].c_str());
        return false;
    }
    std::ifstream file;
    file.open(argv[1], std::ios::binary);
    if (!file.is_open()) {
        ERROR_LOG("Open file failed. path = %s", argv[1].c_str());
        return false;
    }
    std::filebuf *buf = file.rdbuf();
    size_t inputByteSize = buf->pubseekoff(0, std::ios::end, std::ios::in);
    if (size == 0) {
        ERROR_LOG("file size is 0");
        file.close();
        return false;
    }

    // 进行码字写入
    struct stat sBuf;
    int fileStatus = stat(argv[2].data(), &sBuf);
    if (fileStatus == -1) {
        ERROR_LOG("failed to get file");
        return false;
    }
    if (S_ISREG(sBuf.st_mode) == 0) {
        ERROR_LOG("%s is not a file, please enter a file", argv[2].c_str());
        return false;
    }
    std::ifstream ofile;
    ofile.open(argv[1], std::ios::binary);
    if (!ofile.is_open()) {
        ERROR_LOG("Open file failed. path = %s", argv[2].c_str());
        return false;
    }
    std::filebuf *obuf = ofile.rdbuf();

    // NPU 
    //CHECK_ACL(aclInit(nullptr));
    // 获取设备参数
    CHECK_ACL(aclInit("./acl.json"));
    aclrtContext context;
    int32_t deviceId = 0;
    CHECK_ACL(aclrtSetDevice(deviceId));
    CHECK_ACL(aclrtCreateContext(&context, deviceId));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));
    aclrtDeviceProperties deviceProps;
    CHECK_ACL(aclrtGetDeviceProperties(&deviceProps, deviceId));
    int maxComputeUnits = deviceProps.maxComputeUnits;
    // 输出最大Block数目相关信息
    std::cout << "设备名称: " << deviceProps.name << std::endl;
    std::cout << "每个SM的最大Block数目: " << deviceProps.maxBlockPerSM << std::endl;
    std::cout << "SM总数: " << deviceProps.multiProcessorCount << std::endl;
    std::cout << "每个Block的最大线程数: " << deviceProps.maxThreadsPerBlock << std::endl;
    // 计算理论最大Block数目（需根据任务需求调整）
    int maxBlocks = deviceProps.maxBlockPerSM * deviceProps.multiProcessorCount;
    std::cout << "理论最大Block数目（全SM）: " << maxBlocks << std::endl;


    // 分配缓冲区
    uint8_t *srcHost, *compresedHost;
    CHECK_ACL(aclrtMallocHost((void**)(&srcHost), inputByteSize));
    CHECK_ACL(aclrtMallocHost((void**)(&compressedHost), getFinalbufferSize(inputByteSize)));
    uint8_t *srcDevice;
    CHECK_ACL(aclrtMalloc((void**)&srcDevice, inputByteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    uint8_t *tempBuffer;
    CHECK_ACL(aclrtMalloc((void**)&tempBuffer, getMaxtempbufferSize(inputByteSize), ACL_MEM_MALLOC_HUGE_FIRST));
    uint8_t *final;
    CHECK_ACL(aclrtMalloc((void**)&final, getFinalbufferSize(inputByteSize), ACL_MEM_MALLOC_HUGE_FIRST));
    uint8_t *histogramDevice;
    CHECK_ACL(aclrtMalloc((void**)&histogramDevice, blockNum * HISTOGRAM_BINS * sizeof(int), ACL_MEM_MALLOC_HUGE_FIRST));
    uint8_t *compressedSize;
    CHECK_ACL(aclrtMalloc((void**)&compressedSize, blockNum * sizeof(uint32_t), ACL_MEM_MALLOC_HUGE_FIRST));
    uint8_t *compressedSizePrefix;
    CHECK_ACL(aclrtMalloc((void**)&compressedSizePrefix, blockNum * sizeof(uint32_t), ACL_MEM_MALLOC_HUGE_FIRST));

    uint32_t totalCompressedSize;

    buf->pubseekpos(0, std::ios::in);
    buf->sgetn(static_cast<char *>(srcHost), inputByteSize);
    file.close();

    CHECK_ACL(aclrtMemcpy(srcDevice, inputByteSize, srcHost, inputByteSize, ACL_MEMCPY_HOST_TO_DEVICE));

    uint32_t blockNum = (inputByteSize + 4096 - 1) / 4096;
    double comp_time = 0.0;
    for(int i = 0; i < 11; i ++){
        auto start = std::chrono::high_resolution_clock::now();  

        // 压缩开始
        compress(uint32_t blockNum, nullptr, void* stream, uint8_t* srcDevice, uint8_t* tempBuffer, uint8_t* final, int32_t* histogramDevice, uint32_t* compressedSize, uint32_t* compressedSizePrefix, uint32_t totalCompressedSize);
        
        auto end = std::chrono::high_resolution_clock::now();  
        if(i > 5)
            comp_time += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1e3;  
    }
    comp_time /= 5;
    double c_bw = ( 1.0 * fileSize / 1e6 ) / ( (comp_time) * 1e-3 );  
    std::cout << "comp   time " << std::fixed << std::setprecision(3) << comp_time << " ms B/W "   
                  << std::fixed << std::setprecision(1) << c_bw << " MB/s " << std::endl;

    CHECK_ACL(aclrtSynchronizeStream(stream));

    CHECK_ACL(aclrtMemcpy(compressedHost, totalCompressedSize, compressedDevice, totalCompressedSize, ACL_MEMCPY_DEVICE_TO_HOST));
    write(ofile, compressedHost, totalCompressedSize);

    CHECK_ACL(aclrtFree(xDevice));
    CHECK_ACL(aclrtFree(yDevice));
    CHECK_ACL(aclrtFreeHost(xHost));
    CHECK_ACL(aclrtFreeHost(yHost));

    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtDestroyContext(context));
    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclFinalize());
    return 0;
}