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

#ifndef __CCE_KT_TEST__
#include "acl/acl.h"
extern void compress(uint32_t coreDim, void* l2ctrl, void* stream, uint8_t* x, uint8_t* y);
#else
#include "tikicpulib.h"
extern "C" __global__ __aicore__ void compress(GM_ADDR x, GM_ADDR y);
#endif

int main(int32_t argc, char* argv[])
{
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

    // size_t outputByteSize = 8 * 200 * 1024 * sizeof(float);

#ifdef __CCE_KT_TEST__
    // CPU
    uint8_t* src = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* compressed = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    printf("[cpu debug]>>> inputByteSize: %d\n", inputByteSize); 

    // ReadFile(argv[1], inputByteSize, x, inputByteSize);
    buf->pubseekpos(0, std::ios::in);
    buf->sgetn(static_cast<char *>(src), inputByteSize);
    file.close();

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(compress_cpu, BLOCK_NUM, src, compressed); // use this macro for cpu debug
    // WriteFile(argv[2], y, outputByteSize);
    write(ofile, compressed, outputByteSize);
    AscendC::GmFree((void *)src);
    AscendC::GmFree((void *)compressed);
    
#else
   // NPU 
    //CHECK_ACL(aclInit(nullptr));
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


    uint8_t *srcHost, *compresedHost;
    uint8_t *srcDevice, *compressedDevice;
    uint8_t *tempHistogram;
    uint8_t *finalHistogram;
    uint32_t totalCompressedSize;
    CHECK_ACL(aclrtMallocHost((void**)(&srcHost), inputByteSize));
    CHECK_ACL(aclrtMallocHost((void**)(&compressedHost), outputByteSize));
    CHECK_ACL(aclrtMalloc((void**)&srcDevice, inputByteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void**)&compressedDevice, outputByteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void**)&tempHistogram, BLOCK_NUM * kNumSymbols * sizeof(uint8_t), ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void**)&finalHistogram, kNumSymbols * sizeof(uint8_t), ACL_MEM_MALLOC_HUGE_FIRST));

    // ReadFile(argv[1], inputByteSize, xHost, inputByteSize);
    buf->pubseekpos(0, std::ios::in);
    buf->sgetn(static_cast<char *>(srcHost), inputByteSize);
    file.close();

    CHECK_ACL(aclrtMemcpy(srcDevice, inputByteSize, srcHost, inputByteSize, ACL_MEMCPY_HOST_TO_DEVICE));
    double comp_time = 0.0;
    for(int i = 0; i < 11; i ++){
        auto start = std::chrono::high_resolution_clock::now();  
        compress(BLOCK_NUM, nullptr, stream, srcDevice, inputByteSize, compressedDevice, totalCompressedSize);
        auto end = std::chrono::high_resolution_clock::now();  
        if(i > 5)
            comp_time += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1e3;  
    }
    comp_time /= 5;
    double c_bw = ( 1.0 * fileSize / 1e6 ) / ( (comp_time) * 1e-3 );  
    std::cout << "comp   time " << std::fixed << std::setprecision(3) << comp_time << " ms B/W "   
                  << std::fixed << std::setprecision(1) << c_bw << " MB/s " << std::endl;

    CHECK_ACL(aclrtSynchronizeStream(stream));

    CHECK_ACL(aclrtMemcpy(compressedHost, outputByteSize, compressedDevice, outputByteSize, ACL_MEMCPY_DEVICE_TO_HOST));
    // WriteFile(argv[2], compressedHost, outputByteSize);
    write(ofile, compressedHost, outputByteSize);

    CHECK_ACL(aclrtFree(xDevice));
    CHECK_ACL(aclrtFree(yDevice));
    CHECK_ACL(aclrtFreeHost(xHost));
    CHECK_ACL(aclrtFreeHost(yHost));

    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtDestroyContext(context));
    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclFinalize());
#endif
    return 0;
}