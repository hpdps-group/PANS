/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
 * This file constains code of cpu debug and npu code.We read data from bin file
 * and write result to file.
 */

#include "snec_utils.h"
#include "snec_host.h"
extern "C" void enec_table(uint32_t totalUncompressedSize, void *stream, uint8_t *srcDevice, uint8_t *histogramDevice, uint32_t dataType);
extern "C" void enec_compress(Header *cphd, void *stream, uint8_t *srcDevice, uint8_t *compressedDevice, uint8_t *compressedFinal, uint8_t *histogramDevice, uint8_t *blockCompSizeDevice);
extern "C" void enec_merge(Header *cphd, void *stream, uint8_t *compressedDevice, uint8_t *compressedFinal, uint8_t* histogramDevice, uint8_t* blockCompSizeDevice, uint32_t bufferSize);
int main(int32_t argc, char *argv[])
{
    std::string inputFile;
    std::string outputFile;
    size_t inputByteSize = 0;
    int tileLength = 16;
    int dataType = 0;
    int compLevel = 0;
    bool isStatistics = false;
    if (argc < 4){
        std::cerr << "Usage: " << argv[0]
                  << " <input.file> <output.file> <inputByteSize>"
                  << " [tileLength=16] [dataTypes=0] [compLevel=0] [isStatistics=1]\n";
        std::cerr << "\nPositional arguments:\n"
                  << "  1. input.file      : Input file path\n"
                  << "  2. output.file     : Output file path\n"
                  << "  3. inputByteSize   : Size of input data in bytes\n"
                  << "  4. tileLength      : Tile size (default: 16)\n"
                  << "  5. dataTypes       : Data format (0=BF16, 1=FP16, 2=FP32) (default: 0)\n"
                  << "  6. compLevel       : Compression level (0-9) (default: 1)\n"
                  << "  7. isStatistics    : Enable statistics (0=disable, 1=enable) (default: 1)\n";
        return 1;
    }
    inputFile = argv[1];
    outputFile = argv[2];
    inputByteSize = std::stoul(argv[3]);
    if (argc > 4)
        tileLength = std::stoi(argv[4]);
    if (argc > 5)
        dataType = std::stoi(argv[5]);
    if (argc > 6)
        compLevel = std::stoi(argv[6]);
    if (argc > 7)
        isStatistics = std::stoi(argv[7]) != 0;
    ifstream file(inputFile, ios::binary);
    if (!file){
        cerr << "Unable to open the file: " << inputFile << endl;
        return EXIT_FAILURE;
    }
    streamsize fileSize = file.tellg();
    CHECK_ACL(aclInit(nullptr));
    aclrtContext context;
    int32_t deviceId = 0;
    CHECK_ACL(aclrtSetDevice(deviceId));
    CHECK_ACL(aclrtCreateContext(&context, deviceId));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));
    uint16_t *host = (uint16_t *)malloc(inputByteSize);
    file.read(reinterpret_cast<char *>(host), inputByteSize);
    file.close();
    uint32_t DATA_BLOCK_BYTE_NUM_C = DATA_BLOCK_ELEMENT_NUM_C * sizeof(uint16_t);
    uint32_t tileNum = DATA_BLOCK_ELEMENT_NUM_C / tileLength;
    uint8_t *compressedHost;
    CHECK_ACL(aclrtMallocHost((void **)(&compressedHost), getFinalbufferSize(inputByteSize, tileNum, DATA_BLOCK_BYTE_NUM_C)));
    Header *cphd = (Header *)compressedHost;
    cphd->dataBlockSize = DATA_BLOCK_BYTE_NUM_C;
    cphd->dataBlockNum = (inputByteSize + DATA_BLOCK_BYTE_NUM_C - 1) / DATA_BLOCK_BYTE_NUM_C;
    cphd->threadBlockNum = BLOCK_NUM;
    cphd->compLevel = 0;
    cphd->totalUncompressedBytes = inputByteSize;
    cphd->totalCompressedBytes = 0;
    cphd->tileLength = tileLength;
    cphd->dataType = dataType;
    cphd->mblLength = 4;
    cphd->options = 3;
    cphd->histogramBytes = HISTOGRAM_BINS;
    uint8_t *srcDevice, *compressedDevice, *compressedFinal, *histogramDevice, *blockCompSizeDevice;
    CHECK_ACL(aclrtMalloc((void **)&srcDevice, inputByteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&compressedDevice, getFinalbufferSize(inputByteSize, tileNum, DATA_BLOCK_BYTE_NUM_C), ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&compressedFinal, getFinalbufferSize(inputByteSize, tileNum, DATA_BLOCK_BYTE_NUM_C), ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&histogramDevice, BLOCK_NUM * HISTOGRAM_BINS * sizeof(int), ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&blockCompSizeDevice, BLOCK_NUM * 32 * sizeof(uint8_t), ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMemcpy(srcDevice, inputByteSize, host, inputByteSize, ACL_MEMCPY_HOST_TO_DEVICE));
    auto start = std::chrono::high_resolution_clock::now();
    enec_table(inputByteSize, stream, srcDevice, histogramDevice, dataType);
    CHECK_ACL(aclrtSynchronizeStream(stream));
    auto end = std::chrono::high_resolution_clock::now();
    double table_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1e3;
    std::cout << "table   time: " << std::fixed << std::setprecision(3) << table_time << " ms" << std::endl;
    double comp_time = 0.0;
    double time = 0.0;
    for (int i = 0; i < 11; i++){
        auto start = std::chrono::high_resolution_clock::now();
        enec_compress(cphd, stream, srcDevice, compressedDevice, compressedFinal, histogramDevice, blockCompSizeDevice);
        CHECK_ACL(aclrtSynchronizeStream(stream));
        auto end = std::chrono::high_resolution_clock::now();
        if (i > 5)
            comp_time += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1e3;
    }
    comp_time /= 5;
    double c_bw = (1.0 * inputByteSize / 1024 / 1024) / ((comp_time) * 1e-3);
    std::cout << "comp   time " << std::fixed << std::setprecision(3) << comp_time << " ms B/W "
              << std::fixed << std::setprecision(1) << c_bw << " MB/s " << std::endl;
    int datablockNum = (inputByteSize + DATA_BLOCK_BYTE_NUM_C - 1) / DATA_BLOCK_BYTE_NUM_C;
    int datablockNumPerBLOCK = (datablockNum + BLOCK_NUM - 1) / BLOCK_NUM;
    uint32_t bufferSize = (DATA_BLOCK_BYTE_NUM_C / 2 * datablockNumPerBLOCK);
    start = std::chrono::high_resolution_clock::now();
    enec_merge(cphd, stream, compressedDevice, compressedFinal, histogramDevice, blockCompSizeDevice, bufferSize);
    CHECK_ACL(aclrtSynchronizeStream(stream));
    end = std::chrono::high_resolution_clock::now();
    double merge_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1e3;
    std::cout << "merge  time: " << std::fixed << std::setprecision(3) << merge_time << " ms" << std::endl;
    CHECK_ACL(aclrtMemcpy(compressedHost + 32, getFinalbufferSize(inputByteSize, tileNum, DATA_BLOCK_BYTE_NUM_C), compressedDevice + 32, getFinalbufferSize(inputByteSize, tileNum, DATA_BLOCK_BYTE_NUM_C), ACL_MEMCPY_DEVICE_TO_HOST));
    uint8_t *histogramHost;
    CHECK_ACL(aclrtMallocHost((void **)(&histogramHost), BLOCK_NUM * HISTOGRAM_BINS * sizeof(int)));
    CHECK_ACL(aclrtMemcpy(histogramHost, BLOCK_NUM * HISTOGRAM_BINS * sizeof(int), histogramDevice, BLOCK_NUM * HISTOGRAM_BINS * sizeof(int), ACL_MEMCPY_DEVICE_TO_HOST));
    uint8_t *table8 = getTable(cphd, compressedHost);
    uint32_t *hist32 = (uint32_t *)histogramHost;
    for (int i = 0; i < HISTOGRAM_BINS; i++)
        table8[hist32[i] >> 14] = (uint8_t)i;
    uint8_t *blockCompSizeHost;
    CHECK_ACL(aclrtMallocHost((void **)(&blockCompSizeHost), BLOCK_NUM * 8 * sizeof(int)));
    CHECK_ACL(aclrtMemcpy(blockCompSizeHost, BLOCK_NUM * 8 * sizeof(int), blockCompSizeDevice, BLOCK_NUM * 8 * sizeof(int), ACL_MEMCPY_DEVICE_TO_HOST));
    uint32_t totalCompSize = 0;
    uint32_t *compsizePrefix = (uint32_t *)(getCompSizePrefix(cphd, compressedHost));
    compsizePrefix[0] = 0;
    uint32_t *blockCompSizeHost32 = (uint32_t *)blockCompSizeHost;
    totalCompSize = totalCompSize + blockCompSizeHost32[0];
    for (int i = 1; i < BLOCK_NUM; i++){
        compsizePrefix[i] = compsizePrefix[i - 1] + blockCompSizeHost32[(i - 1) * 8];
        totalCompSize += blockCompSizeHost32[i * 8];
    }
    uint8_t *compexpHostStart = getCompressed_exp(cphd, compressedHost);
    uint8_t *compexpDeviceStart = getCompressed_exp(cphd, compressedDevice);
    for (int i = 0; i < BLOCK_NUM; i++){
        uint8_t *comphstart = compexpHostStart + compsizePrefix[i];
        uint8_t *compdstart = compexpDeviceStart + ((i + 1) * bufferSize - blockCompSizeHost32[i * 8]);
        CHECK_ACL(aclrtMemcpy(comphstart, blockCompSizeHost32[i * 8], compdstart, blockCompSizeHost32[i * 8], ACL_MEMCPY_DEVICE_TO_HOST));
    }
    CHECK_ACL(aclrtMemcpy(getMsdata(cphd, compressedHost), inputByteSize / 2 + cphd->dataBlockNum * (cphd->dataBlockSize / (cphd->tileLength * sizeof(uint16_t))) / 2, getMsdata(cphd, compressedFinal), inputByteSize / 2 + cphd->dataBlockNum * (cphd->dataBlockSize / (cphd->tileLength * sizeof(uint16_t))) / 2, ACL_MEMCPY_DEVICE_TO_HOST));
    uint32_t totalCompressedSize = 0;
    if (cphd->dataType == 0 | cphd->dataType == 1){
        totalCompressedSize = 32 + HISTOGRAM_BINS + inputByteSize / 2 + cphd->dataBlockNum * (cphd->dataBlockSize / (cphd->tileLength * sizeof(uint16_t))) / 2 + BLOCK_NUM * 4 + totalCompSize;
    }
    else{
        totalCompressedSize = 32 + HISTOGRAM_BINS + inputByteSize / 2 + cphd->dataBlockNum * (cphd->dataBlockSize / (cphd->tileLength * sizeof(float))) / 2 + BLOCK_NUM * 4 + totalCompSize;
    }
    cphd->totalCompressedBytes = totalCompressedSize;
    uint8_t *compressedHostMerged;
    CHECK_ACL(aclrtMallocHost((void **)(&compressedHostMerged), getFinalbufferSize(inputByteSize, tileNum, DATA_BLOCK_BYTE_NUM_C)));
    CHECK_ACL(aclrtMemcpy(compressedHostMerged, getFinalbufferSize(inputByteSize, tileNum, DATA_BLOCK_BYTE_NUM_C), compressedFinal, getFinalbufferSize(inputByteSize, tileNum, DATA_BLOCK_BYTE_NUM_C), ACL_MEMCPY_DEVICE_TO_HOST));
    Header *cphdM = (Header *)compressedHostMerged;
    uint32_t totalCompressedSize0 = cphdM->totalCompressedBytes;
    printf("blockNum: %d\n", cphdM->dataBlockNum);
    printf("threadBlockNum: %d\n", cphdM->threadBlockNum);
    printf("compLevel: %d\n", cphdM->compLevel);
    printf("tileLength: %d\n", cphdM->tileLength);
    printf("dataType: %d\n", cphdM->dataType);
    printf("mblLength: %d\n", cphdM->mblLength);
    printf("options: %d\n", cphdM->options);
    printf("histogramBytes: %d\n", cphdM->histogramBytes);
    printf("Size before compression: %d\n", cphdM->totalUncompressedBytes);
    printf("Compressed size: %d\n", totalCompressedSize0);
    printf("cr: %f\n", computeCr(inputByteSize, totalCompressedSize0));
    for(int i = 0; i < totalCompressedSize0; i ++){
        if(compressedHost[i] != compressedHostMerged[i]){
            std::cout << "compressedHost and compressedHostMerged are not equal at index " << i << std::endl;
            break;
        }
    }
    std::cout << "merge B/W: " << std::fixed << std::setprecision(1) << (1.0 * totalCompressedSize / 1024 / 1024) / ((merge_time) * 1e-3) << " MB/s" << std::endl;
    std::ofstream ofile;
    ofile.open(outputFile, std::ios::binary);
    std::filebuf *obuf = ofile.rdbuf();
    ofile.write(reinterpret_cast<char *>(compressedHostMerged), totalCompressedSize);
    ofile.close();
    free(host);
    CHECK_ACL(aclrtFree(srcDevice));
    CHECK_ACL(aclrtFree(compressedDevice));
    CHECK_ACL(aclrtFree(histogramDevice));
    CHECK_ACL(aclrtFree(blockCompSizeDevice));
    CHECK_ACL(aclrtFreeHost(histogramHost));
    CHECK_ACL(aclrtFreeHost(compressedHost));
    CHECK_ACL(aclrtFreeHost(compressedHostMerged));
    CHECK_ACL(aclrtFreeHost(blockCompSizeHost));
    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclFinalize());
    return 0;
}