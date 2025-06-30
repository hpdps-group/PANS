/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
 * This file constains code of cpu debug and npu code.We read data from bin file
 * and write result to file.
 */

#include "snec_utils.h"
#include "snec_host.h"

extern "C" void decompress(Header *cphd, void *stream, uint8_t *compressed, uint8_t *decompressed);
extern "C" void verify(Header *cphd, void *stream, uint8_t *compressed, uint8_t *source, uint8_t *out);

int main(int32_t argc, char *argv[])
{
    std::string inputFile;
    std::string outputFile;
    std::string sourceFile;

    if (argc < 4)
    {
        std::cerr << "Usage: " << argv[0]
                  << " <input.file> <output.file> ";
        std::cerr << "\nPositional arguments:\n"
                  << "  1. input.file      : Input file path\n"
                  << "  2. output.file     : Output file path\n"
                  << "  3. source.file     : Source file path\n";
        return 1;
    }

    inputFile = argv[1];
    outputFile = argv[2];
    sourceFile = argv[3];

    ifstream file(inputFile, ios::binary);
    if (!file)
    {
        cerr << "Unable to open the file: " << inputFile << endl;
        return EXIT_FAILURE;
    }
    file.seekg(0, ios::end);
    streamsize fileSize = file.tellg();
    file.seekg(0, ios::beg);
    CHECK_ACL(aclInit(nullptr));
    aclrtContext context;
    int32_t deviceId = 0;
    CHECK_ACL(aclrtSetDevice(deviceId));
    CHECK_ACL(aclrtCreateContext(&context, deviceId));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    uint8_t *compressed = (uint8_t *)malloc(fileSize);
    file.read(reinterpret_cast<char *>(compressed), fileSize);
    file.close();

    Header *cphd = reinterpret_cast<Header *>(compressed);

    printf("tileLength: %d\n", cphd->tileLength);
    printf("totalUncompressedBytes: %d\n", cphd->totalUncompressedBytes);
    printf("fileSize: %d\n", fileSize);
    printf("dataBlockSize: %d\n", cphd->dataBlockSize);
    printf("dataBlockNum: %d\n", cphd->dataBlockNum);
    printf("threadBlockNum: %d\n", cphd->threadBlockNum);
    printf("dataType: %d\n", cphd->dataType);
    uint32_t *prefix = (uint32_t *)getCompSizePrefix(cphd, compressed);

    uint8_t *compressedDevice, *decompressed;
    CHECK_ACL(aclrtMalloc((void **)&compressedDevice, fileSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&decompressed, cphd->totalUncompressedBytes, ACL_MEM_MALLOC_HUGE_FIRST));

    CHECK_ACL(aclrtMemcpy(compressedDevice, fileSize, compressed, fileSize, ACL_MEMCPY_HOST_TO_DEVICE));

    double decomp_time = 0.0;
    double time = 0.0;
    for (int i = 0; i < 11; i++)
    {
        auto start = std::chrono::high_resolution_clock::now();
        decompress(cphd, stream, compressedDevice, decompressed);
        CHECK_ACL(aclrtSynchronizeStream(stream));
        auto end = std::chrono::high_resolution_clock::now();
        if (i > 5)
            decomp_time += // time;
                std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1e3;
    }
    decomp_time /= 5;
    double c_bw = (1.0 * cphd->totalUncompressedBytes / 1024 / 1024) / ((decomp_time) * 1e-3);
    std::cout << "decomp   time " << std::fixed << std::setprecision(3) << decomp_time << " ms B/W "
              << std::fixed << std::setprecision(1) << c_bw << " MB/s " << std::endl;

    uint8_t *decompressedHost;
    CHECK_ACL(aclrtMallocHost((void **)(&decompressedHost), cphd->totalUncompressedBytes));
    CHECK_ACL(aclrtMemcpy(decompressedHost, cphd->totalUncompressedBytes, decompressed, cphd->totalUncompressedBytes, ACL_MEMCPY_DEVICE_TO_HOST));

    std::ofstream ofile;
    ofile.open(outputFile, std::ios::binary);

    aclrtStream stream0 = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream0));
    std::filebuf *obuf = ofile.rdbuf();
    ofile.write(reinterpret_cast<char *>(decompressedHost), cphd->totalUncompressedBytes);
    ofile.close();

    ifstream file0(sourceFile, ios::binary);
    if (!file0)
    {
        cerr << "Unable to open the file: " << sourceFile << endl;
        return EXIT_FAILURE;
    }
    uint8_t *source = (uint8_t *)malloc(cphd->totalUncompressedBytes);
    file0.read(reinterpret_cast<char *>(source), cphd->totalUncompressedBytes);
    file0.close();

    uint8_t *srcDevice, *outDevice;
    CHECK_ACL(aclrtMalloc((void **)&srcDevice, cphd->totalUncompressedBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&outDevice, cphd->dataBlockNum * 32, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMemcpy(srcDevice, cphd->totalUncompressedBytes, source, cphd->totalUncompressedBytes, ACL_MEMCPY_HOST_TO_DEVICE));

    for (int i = 0; i < cphd->totalUncompressedBytes; i++)
    {
        if (source[i] != decompressedHost[i])
        {
            int blockid = i / cphd->dataBlockSize;
            printf("fatal block id: %d, num: %d, decompressed: %d, source: %d\n", blockid, i % cphd->dataBlockSize, decompressedHost[i], source[i]);
            break;
        }
    }

    // verify(cphd, stream0, decompressed, srcDevice, outDevice);
    // CHECK_ACL(aclrtSynchronizeStream(stream0));
    // uint8_t *outHost = (uint8_t*)malloc(cphd->dataBlockNum * 32);
    // CHECK_ACL(aclrtMemcpy(outHost, cphd->dataBlockNum * 32, outDevice, cphd->dataBlockNum * 32, ACL_MEMCPY_DEVICE_TO_HOST));
    // uint16_t* out16 = (uint16_t*)outHost;
    // // for(int i = 0; i < 32; i ++){
    // //     printf("%d ",source[i]);
    // // }
    // // for(int i = 0; i < cphd->dataBlockNum; i ++){
    // //     if(out16[i * 16] != 0){
    // //         printf("block %d decomp %d!\n", i, out16[i * 16]);
    // //     }
    // // }

    CHECK_ACL(aclrtFree(srcDevice));
    CHECK_ACL(aclrtFree(decompressed));

    CHECK_ACL(aclrtFreeHost(decompressedHost));

    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclFinalize());

    return 0;
}