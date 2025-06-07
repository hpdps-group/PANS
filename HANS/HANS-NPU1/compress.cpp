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
#include <bitset>
#include <chrono>

#include "acl/acl.h"
#include "acl/acl_op_compiler.h"
// #include "graph/tensor.h"
// #include "graph/types.h"
// #include "graph/operator_factory.h"
#include "tiling/platform/platform_ascendc.h"

using namespace std;

constexpr uint32_t DATA_BLOCK_BYTE_NUM = 4096;// 单位为字节
constexpr int32_t BUFFER_NUM = 2; // 双缓冲
constexpr int32_t BLOCK_NUM = 256;// block的数量
constexpr uint32_t HISTOGRAM_BINS = 256;// 尽可能是2的幂，直方图桶数
constexpr uint32_t TILE_LEN = 16; // 每个Tile处理32个单元(单元指输入数据的类型)
constexpr uint32_t TILE_NUM = DATA_BLOCK_BYTE_NUM / sizeof(uint32_t) / TILE_LEN; // 每个数据块包含TILE_NUM个TILE

constexpr uint32_t NUM = 64;
constexpr uint32_t TILE_LEN_E = TILE_LEN * NUM;// 16 * 64 = 1024, TILE_LEN_E代表在eh内核中一次处理的元素数量
constexpr uint32_t TILE_NUM_E = DATA_BLOCK_BYTE_NUM / sizeof(uint32_t) / TILE_LEN_E;

#define CHECK_ACL(x)                                                                        \
    do {                                                                                    \
        aclError __ret = x;                                                                 \
        if (__ret != ACL_ERROR_NONE) {                                                      \
            std::cerr << __FILE__ << ":" << __LINE__ << " aclError:" << __ret << std::endl; \
        }                                                                                   \
    } while (0);

#define INFO_LOG(fmt, args...) fprintf(stdout, "[INFO]  " fmt "\n", ##args)
#define WARN_LOG(fmt, args...) fprintf(stdout, "[WARN]  " fmt "\n", ##args)
#define ERROR_LOG(fmt, args...) fprintf(stdout, "[ERROR]  " fmt "\n", ##args)

extern "C" void table(uint32_t datablockNum, void* stream, uint8_t* srcDevice, uint8_t* tempBuffer, uint8_t* final, int32_t* histogramDevice, uint32_t* tilePrefix, uint32_t* compressedSize, uint32_t* compressedSizePrefix, uint32_t totalUncompressedSize, uint32_t tempSize);
extern "C" double compress(uint32_t datablockNum, void* stream, uint8_t* srcDevice, uint8_t* tempBuffer, uint8_t* final, int32_t* histogramDevice, uint32_t* tilePrefix, uint32_t* compressedSize, uint32_t* compressedSizePrefix, uint32_t totalUncompressedSize);
extern "C" void merge(uint32_t datablockNum, void* stream, uint8_t* srcDevice, uint8_t* tempBuffer, uint8_t* final, int32_t* histogramDevice, uint32_t* tilePrefix, uint32_t* compressedSize, uint32_t* compressedSizePrefix, uint32_t totalUncompressedSize);

// 一个block每次处理32次32个uint32_t(2个uint16_t组成一个uint32_t)，即32 * 32 * 4 = 4096字节数据，每个4096字节数据块产生的max_bits_length数组大小为32字节

int getFinalbufferSize(uint32_t byteSize){
    // int FinalBufferSize = 0;
    int datablockNum = (byteSize + DATA_BLOCK_BYTE_NUM - 1) / DATA_BLOCK_BYTE_NUM;
    int datablockNumPerBLOCK = (datablockNum + BLOCK_NUM - 1) / BLOCK_NUM;
    int FinalBufferSize = 32 + HISTOGRAM_BINS * sizeof(uint8_t) + TILE_NUM * datablockNum +  DATA_BLOCK_BYTE_NUM / 2 * datablockNum + (DATA_BLOCK_BYTE_NUM / 2 * datablockNumPerBLOCK) * BLOCK_NUM * 2;
    return FinalBufferSize;
}

int main(int32_t argc, char* argv[])
{
    // 获取输入参数
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <input.file> <output.file> <inputByteSize>" << std::endl;
        return 1;
    }
    // auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    // auto aivNum = ascendcPlatform.GetCoreNumAiv();
    // printf("aivNum: %d\n", aivNum);

    ifstream file(argv[1], ios::binary);
    if (!file) {
        cerr << "无法打开文件: " << argv[1] << endl;
        return EXIT_FAILURE;
    }
    streamsize fileSize = file.tellg();

    // 进行码字写入
    std::ofstream ofile;
    ofile.open(argv[2], std::ios::binary);

    std::filebuf *obuf = ofile.rdbuf();

    CHECK_ACL(aclInit(nullptr));
    aclrtContext context;
    int32_t deviceId = 0;
    CHECK_ACL(aclrtSetDevice(deviceId));
    CHECK_ACL(aclrtCreateContext(&context, deviceId));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    // int datablockNum = (inputByteSize + 4096 - 1) / 4096;
    // 分配缓冲区
    // uint16_t host[DATA_BLOCK_BYTE_NUM * 1024];
    uint16_t* host = NULL;

    int inputByteSize = atoi(argv[3]);
    // DATA_BLOCK_BYTE_NUM * sizeof(uint16_t) / 2;
    host = (uint16_t*)malloc(inputByteSize);
    int datablockNum = (inputByteSize + DATA_BLOCK_BYTE_NUM - 1) / DATA_BLOCK_BYTE_NUM;
    uint8_t *srcHost, *compressedHost, *histogramHost, *bufferHost, *compressedSizeHost, *tilePrefixHost, *compressedPrefixHost;
    CHECK_ACL(aclrtMallocHost((void**)(&srcHost), inputByteSize));
    CHECK_ACL(aclrtMallocHost((void**)(&compressedHost), getFinalbufferSize(inputByteSize)));
    CHECK_ACL(aclrtMallocHost((void**)(&histogramHost), HISTOGRAM_BINS * sizeof(int)));
    CHECK_ACL(aclrtMallocHost((void**)(&bufferHost), datablockNum * DATA_BLOCK_BYTE_NUM * 2));
    CHECK_ACL(aclrtMallocHost((void**)(&compressedSizeHost), datablockNum * sizeof(uint32_t)));
    CHECK_ACL(aclrtMallocHost((void**)(&tilePrefixHost), datablockNum * TILE_NUM * sizeof(uint32_t)));
    CHECK_ACL(aclrtMallocHost((void**)(&compressedPrefixHost), datablockNum * TILE_NUM * sizeof(uint32_t)));
    // printf("1 success!\n");
    uint8_t *srcDevice;
    CHECK_ACL(aclrtMalloc((void**)&srcDevice, inputByteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    uint8_t *tempBuffer;
    CHECK_ACL(aclrtMalloc((void**)&tempBuffer, datablockNum * DATA_BLOCK_BYTE_NUM * 2, ACL_MEM_MALLOC_HUGE_FIRST));
    uint8_t *final;
    CHECK_ACL(aclrtMalloc((void**)&final, getFinalbufferSize(inputByteSize), ACL_MEM_MALLOC_HUGE_FIRST));
    int32_t *histogramDevice;
    CHECK_ACL(aclrtMalloc((void**)&histogramDevice, BLOCK_NUM * HISTOGRAM_BINS * sizeof(int), ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMemset((void*)histogramDevice, BLOCK_NUM * HISTOGRAM_BINS * sizeof(int), 0, BLOCK_NUM * HISTOGRAM_BINS * sizeof(int)));
    uint32_t *tilePrefix;
    CHECK_ACL(aclrtMalloc((void**)&tilePrefix, datablockNum * TILE_NUM * sizeof(uint32_t), ACL_MEM_MALLOC_HUGE_FIRST)); 
    uint32_t *compressedSize;
    CHECK_ACL(aclrtMalloc((void**)&compressedSize, datablockNum * sizeof(uint32_t), ACL_MEM_MALLOC_HUGE_FIRST));
    uint32_t *compressedSizePrefix;
    CHECK_ACL(aclrtMalloc((void**)&compressedSizePrefix, datablockNum * sizeof(uint32_t), ACL_MEM_MALLOC_HUGE_FIRST));
    // printf("2 success!\n");
    // printf("inputByteSize: %d\n", inputByteSize);

    file.read(reinterpret_cast<char*>(host), inputByteSize);//全部按照uint8_t读入
    // ReadFile(argv[1], inputByteSize, (void*)host, inputByteSize);

    // for(int i = 0; i < inputByteSize / 2; i ++){
    //     host[i] = static_cast<uint16_t>(rand() % 65536);
    //     // printf("host[%d]: %d\n", i, host[i]);
    // }
    auto host32 = (uint32_t*)host;
    for(int i = 1000; i < 1024; i ++)
    {
        std::string binary_str = std::bitset<32>(host32[i]).to_string();
        std::string formatted_str;
        for (int j = 0; j < 32; j++) {
            if (j > 0 && j % 4 == 0) {
                formatted_str += " ";
            }
            formatted_str += binary_str[j];
        }
        printf("host32[%d]: %u %s\n", i, host32[i], formatted_str.c_str());
    }

    // srcHost[200] = 0xFF;  // 测试写入
    // printf("Test value: %d\n", srcHost[200]);  // 应输出255
    file.close();
    CHECK_ACL(aclrtMemcpy(srcDevice, inputByteSize, host, inputByteSize, ACL_MEMCPY_HOST_TO_DEVICE));

    // std::vector<int64_t> shape_vec = {TILE_LEN_E * 2};
    // ge::Shape shape(shape_vec);
    uint32_t maxValue = TILE_LEN_E * 2;
    // uint32_t minValue = 0;
    // GetReduceXorSumMaxMinTmpSize(shape, (uint32_t)sizeof(int16_t), false, maxValue, minValue);//获取ReduceXorSum中间temp Size
    auto start = std::chrono::high_resolution_clock::now();  
    table(datablockNum, stream, srcDevice, tempBuffer, final, histogramDevice, tilePrefix, compressedSize, compressedSizePrefix, inputByteSize, maxValue);
    CHECK_ACL(aclrtSynchronizeStream(stream));
    auto end = std::chrono::high_resolution_clock::now();  
    double table_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1e3;
    std::cout << "table   time " << std::fixed << std::setprecision(3) << table_time << std::endl;
    double comp_time = 0.0;
    double time = 0.0;
    for(int i = 0; i < 11; i ++){
        auto start = std::chrono::high_resolution_clock::now();  

        // 压缩开始  
        time  = compress(datablockNum, stream, srcDevice, tempBuffer, final, histogramDevice, tilePrefix, compressedSize, compressedSizePrefix, inputByteSize);
        CHECK_ACL(aclrtSynchronizeStream(stream));
        auto end = std::chrono::high_resolution_clock::now();  
        if(i > 5)
            comp_time += //time;
            std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1e3;  
    }
    // merge(datablockNum, stream, srcDevice, tempBuffer, final, histogramDevice, tilePrefix, compressedSize, compressedSizePrefix, inputByteSize);
    comp_time /= 5;
    double c_bw = ( 1.0 * inputByteSize / 1e6 ) / ( (comp_time) * 1e-3 );  
    std::cout << "comp   time " << std::fixed << std::setprecision(3) << comp_time << " ms B/W "   
                  << std::fixed << std::setprecision(1) << c_bw << " MB/s " << std::endl;

    CHECK_ACL(aclrtMemcpy(compressedHost, getFinalbufferSize(inputByteSize), final, getFinalbufferSize(inputByteSize), ACL_MEMCPY_DEVICE_TO_HOST));
    CHECK_ACL(aclrtMemcpy(histogramHost, HISTOGRAM_BINS * sizeof(int32_t), histogramDevice + 0 * HISTOGRAM_BINS, HISTOGRAM_BINS * sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST));
    CHECK_ACL(aclrtMemcpy(bufferHost, datablockNum * DATA_BLOCK_BYTE_NUM * 2, tempBuffer, datablockNum * DATA_BLOCK_BYTE_NUM * 2, ACL_MEMCPY_DEVICE_TO_HOST));
    CHECK_ACL(aclrtMemcpy(compressedSizeHost, BLOCK_NUM * sizeof(uint32_t), compressedSize, datablockNum * sizeof(uint32_t), ACL_MEMCPY_DEVICE_TO_HOST));
    CHECK_ACL(aclrtMemcpy(tilePrefixHost, datablockNum * TILE_NUM * sizeof(uint32_t), tilePrefix, datablockNum * TILE_NUM * sizeof(uint32_t), ACL_MEMCPY_DEVICE_TO_HOST));
    CHECK_ACL(aclrtMemcpy(compressedPrefixHost, BLOCK_NUM * TILE_NUM * sizeof(uint32_t), compressedSizePrefix, datablockNum * TILE_NUM * sizeof(uint32_t), ACL_MEMCPY_DEVICE_TO_HOST));

    // for(int i = 0; i < 256; i ++)
    // printf("histo[%d]: %d\n", i, histogramHost[i]);
    auto buffer32 = (int32_t*)bufferHost;
    for(int i = 2000; i < 2048; i ++){
        std::string binary_str = std::bitset<32>(buffer32[i]).to_string();
        std::string formatted_str;
        for (int j = 0; j < 32; j++) {
            if (j > 0 && j % 4 == 0) {
                formatted_str += " ";
            }
            formatted_str += binary_str[j];
        }
        printf("buffer32[%d]: %d %s\n", i, buffer32[i], formatted_str.c_str());
    }

    int32_t* hist;
    CHECK_ACL(aclrtMallocHost((void**)(&hist), 256 * sizeof(int32_t)));
    for(int i = 0; i < HISTOGRAM_BINS; i ++)
    {
        hist[i] = 0;
    }
    for(int i = 
    0
    // 1024 * 1
    ; i < datablockNum * 4096 * 2 / 4
    // 1024 * 256
    ; i ++){
        int a = buffer32[i];
        hist[a]++;
    }
    int sum = 0;
    for(int i = 0; i < 256; i ++){
        sum += hist[i];
    }
    // assert(sum == 4096);

    auto ms32 = (uint32_t*)(compressedHost + 32 + HISTOGRAM_BINS * sizeof(uint8_t) + TILE_NUM * datablockNum);
    for(int i = 500; i < 512; i ++){
        std::string binary_str = std::bitset<32>(ms32[i]).to_string();
        std::string formatted_str;
        for (int j = 0; j < 32; j++) {
            if (j > 0 && j % 4 == 0) {
                formatted_str += " ";
            }
            formatted_str += binary_str[j];
        }
        printf("ms32[%d]: %d %s\n", i, ms32[i], formatted_str.c_str());
    }

    int num = 1;
    auto histogram32 = (int32_t*)histogramHost;
    for(int i = 0; i < 256; i ++){
        printf("histo[%d]/hist: %d/%d | ", i, histogram32[i], hist[i]);
        if(num % 8 ==0)
        printf("\n");
        num ++;
    }

    // auto table = (uint8_t*)(compressedHost + 32);
    // for(int i = 0; i < 256; i ++){
    //     printf("%d: %d ", i, (int32_t)table[i]);
    // }
    // printf("\n");

    // auto mbl32 = (uint32_t*)(compressedHost + 32 + HISTOGRAM_BINS * sizeof(uint8_t));
    // for(int i = 0; i < 128; i ++){
    //     std::string binary_str = std::bitset<32>(mbl32[i]).to_string();
    //     std::string formatted_str;
    //     for (int j = 0; j < 32; j++) {
    //         if (j > 0 && j % 4 == 0) {
    //             formatted_str += " ";
    //         }
    //         formatted_str += binary_str[j];
    //     }
    //     printf("mbl32[%d]: %d %s\n", i, mbl32[i], formatted_str.c_str());
    // }

    // auto compressed32 = (uint32_t*)(compressedHost + 32 + HISTOGRAM_BINS * sizeof(uint8_t) + TILE_NUM * datablockNum + 2048 * datablockNum * 2);
    // for(int i = 512+200; i < 512+230; i ++){
    //     std::string binary_str = std::bitset<32>(compressed32[i]).to_string();
    //     std::string formatted_str;
    //     for (int j = 0; j < 32; j++) {
    //         if (j > 0 && j % 4 == 0) {
    //             formatted_str += " ";
    //         }
    //         formatted_str += binary_str[j];
    //     }
    //     printf("compressed32[%d]: %d %s\n", i, compressed32[i], formatted_str.c_str());
    // }

    auto compressedSizeHost32 = (uint32_t*)compressedSizeHost;
    // for(int i = 0; i < 32; i ++){
    //     printf("compressedSize[%d]: %d\n", i, compressedSizeHost32[i]);
    // }


    // auto tilePrefixHost32 = (uint32_t*)tilePrefixHost;
    // for(int i = 0; i < 64; i ++)
    // {
    //     printf("tilePrefix[%d]: %d ", i, tilePrefixHost32[i]);
    // }

    auto compressedPrefixHost32 = (uint32_t*)compressedPrefixHost;
    // for(int i = 0; i < 32; i ++)
    // {
    //     printf("compressedPrefixHost[%d]: %d ", i, compressedPrefixHost32[i]);
    // }
    // printf("\n");

    printf("压缩前大小：%d\n", inputByteSize);
    printf("压缩后大小：%d\n", 32 + HISTOGRAM_BINS * sizeof(uint8_t) + TILE_NUM * datablockNum + 2048 * datablockNum + compressedPrefixHost32[datablockNum - 1] + compressedSizeHost32[datablockNum - 1]);
    printf("cr: %f\n", (float)inputByteSize / (float)(32 + HISTOGRAM_BINS * sizeof(uint8_t) + TILE_NUM * datablockNum + 2048 * datablockNum + compressedPrefixHost32[datablockNum - 1] + compressedSizeHost32[datablockNum - 1]));
    // printf("compressedPrefixHost32[datablockNum - 1] + compressedSizeHost32[datablockNum - 1]: %d\n", compressedPrefixHost32[datablockNum - 1] + compressedSizeHost32[datablockNum - 1]);

    // ofile.write(reinterpret_cast<char*>(compressedHost), totalCompressedSize);
    ofile.close();

    CHECK_ACL(aclrtFree(srcDevice));
    CHECK_ACL(aclrtFree(tempBuffer));
    CHECK_ACL(aclrtFree(final));
    CHECK_ACL(aclrtFree(histogramDevice));
    CHECK_ACL(aclrtFree(tilePrefix));
    CHECK_ACL(aclrtFree(compressedSize));
    CHECK_ACL(aclrtFree(compressedSizePrefix));

    CHECK_ACL(aclrtFreeHost(srcHost));
    CHECK_ACL(aclrtFreeHost(compressedHost));
    CHECK_ACL(aclrtFreeHost(histogramHost));
    CHECK_ACL(aclrtFreeHost(bufferHost));
    CHECK_ACL(aclrtFreeHost(compressedSizeHost));
    CHECK_ACL(aclrtFreeHost(tilePrefixHost));
    CHECK_ACL(aclrtFreeHost(compressedPrefixHost));

    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclFinalize());
    return 0;
}