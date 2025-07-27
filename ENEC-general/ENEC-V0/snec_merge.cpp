/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
 * This file constains code of cpu debug and npu code.We read data from bin file
 * and write result to file.
 */

#include "snec_utils.h"
#include "snec_device.h"

template<typename T>
class mergeKernel {
public:
    __aicore__ inline mergeKernel() {}
    __aicore__ inline void Init(TPipe* pipe, __gm__ uint8_t* finalHeader, __gm__ uint8_t* finalTable, __gm__ uint8_t* finalMs, __gm__ uint8_t* finalMbl, __gm__ uint8_t* finalCompPrefix, __gm__ uint8_t* compedExp, __gm__ uint8_t* finalExp, __gm__ uint8_t* histogramDevice, __gm__ uint8_t* blockCompSize, uint32_t dataBlockSize, uint32_t dataBlockNum, uint32_t threadBlockNum, uint32_t compLevel, uint32_t totalUncompressedBytes, uint32_t totalCompressedBytes, uint32_t tileLength, uint32_t dataType, uint32_t mblLength, uint32_t options, uint32_t histogramBytes, uint32_t bufferSize) {
        this->pipe = pipe;
        this->blockId = GetBlockIdx();
        this->blockNum = GetBlockNum();
        this->datablocksize = dataBlockSize;
        this->datablocknum = dataBlockNum;
        this->threadblocknum = threadBlockNum;
        this->complevel = compLevel;
        this->totaluncompressedbytes = totalUncompressedBytes;
        this->totalcompressedbytes = totalCompressedBytes;
        this->tilelength = tileLength;
        this->datatype = dataType;
        this->mbllength = mblLength;
        this->options = options;
        this->histogrambytes = histogramBytes;
        this->buffersize = bufferSize;
        finalheader.SetGlobalBuffer((__gm__ T*)(finalHeader));
        finaltable.SetGlobalBuffer((__gm__ uint8_t*)(finalTable));
        finalms.SetGlobalBuffer((__gm__ T*)(finalMs));
        finalmbl.SetGlobalBuffer((__gm__ T*)(finalMbl));
        finalcompprefix.SetGlobalBuffer((__gm__ T*)(finalCompPrefix));
        compedexp.SetGlobalBuffer((__gm__ uint8_t*)(compedExp));
        finalexp.SetGlobalBuffer((__gm__ uint8_t*)(finalExp));
        histogram.SetGlobalBuffer((__gm__ T*)(histogramDevice));
        blockcompsize.SetGlobalBuffer((__gm__ T*)(blockCompSize));
    }
public:
    __aicore__ inline void Process(){
        pipe->InitBuffer(compsize, BLOCK_NUM * sizeof(T));
        LocalTensor<T> compsizeLocal = compsize.Get<T>();
        pipe->InitBuffer(compsizeprefix, BLOCK_NUM * sizeof(T));
        LocalTensor<T> compsizeprefixLocal = compsizeprefix.Get<T>();
        pipe->InitBuffer(temp, 32 * 1024);
        LocalTensor<T> tempLocal = temp.Get<T>();
        pipe->InitBuffer(datacopy, 128 * 1024);
        LocalTensor<uint8_t> datacopyLocal = datacopy.Get<uint8_t>();
        DataCopy(tempLocal, blockcompsize, BLOCK_NUM * 32 / sizeof(T));
        int tempsize = tempLocal(0);
        int presize = tempsize;
        uint32_t totalcompsize = tempsize;
        compsizeLocal(0) = tempsize;
        compsizeprefixLocal(0) = 0;
        for(int i = 1; i < BLOCK_NUM; i ++){
            tempsize = tempLocal(i * 8);
            compsizeLocal(i) = tempsize;
            compsizeprefixLocal(i) = compsizeprefixLocal(i - 1) + presize;
            presize = tempsize;
            totalcompsize += tempsize;
        }
        uint32_t thisblocksize = compsizeLocal(blockId);
        uint32_t thisblockprefix = compsizeprefixLocal(blockId);
        uint32_t cyclenum = (thisblocksize + 128 * 1024 - 1) / 128 / 1024;
        uint32_t copystart0 = (blockId + 1) * buffersize - thisblocksize;
        uint32_t copyend0 = thisblockprefix;
        for(int i = 0; i < cyclenum; i ++){
            uint32_t offset = i * 128 * 1024;
            if(offset >= totalcompsize) break;
            uint32_t size = (offset + 128 * 1024 > thisblocksize) ? (thisblocksize - offset) : 128 * 1024;
            uint32_t copystart1 = copystart0 + offset;
            uint32_t copyend1 = copyend0 + offset;
            PipeBarrier<PIPE_ALL>();
            DataCopy(datacopyLocal, compedexp[copystart1], size);
            int32_t eventIDMTE2ToMTE3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
            PipeBarrier<PIPE_ALL>();
            SetFlag<HardEvent::MTE2_MTE3>(eventIDMTE2ToMTE3);
            WaitFlag<HardEvent::MTE2_MTE3>(eventIDMTE2ToMTE3);
            DataCopy(finalexp[copyend1], datacopyLocal, size);
            int32_t eventIDMTE3ToS = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
            PipeBarrier<PIPE_ALL>();
            SetFlag<HardEvent::MTE3_S>(eventIDMTE3ToS);
            WaitFlag<HardEvent::MTE3_S>(eventIDMTE3ToS);
        }
        uint32_t totalcompressedsize = 0;
        if (datatype == 0 | datatype == 1)
            totalcompressedsize = 32 + HISTOGRAM_BINS + totaluncompressedbytes / 2 + datablocknum * (datablocksize / (tilelength * sizeof(uint16_t))) / 2 + threadblocknum * 4 + totalcompsize;
        else
            totalcompressedsize = 32 + HISTOGRAM_BINS + totaluncompressedbytes / 2 + datablocknum * (datablocksize / (tilelength * sizeof(float))) / 2 + threadblocknum * 4 + totalcompsize;

        if(blockId == 0){
            DataCopy(finalcompprefix, compsizeprefixLocal, threadblocknum);
            tempLocal[48](0) = datablocksize;
            tempLocal[48](1) = datablocknum;
            tempLocal[48](2) = threadblocknum | (complevel << 16);
            tempLocal[48](3) = totaluncompressedbytes;
            tempLocal[48](4) = totalcompressedsize;
            tempLocal[48](5) = tilelength | (datatype << 16);
            tempLocal[48](6) = mbllength | (options << 16);
            tempLocal[48](7) = histogrambytes;
            DataCopy(finalheader, tempLocal[48], 8);
            DataCopy(tempLocal[56], histogram, HISTOGRAM_BINS);
            pipe->InitBuffer(table8, HISTOGRAM_BINS);
            LocalTensor<uint8_t> table8Local = table8.Get<uint8_t>();
            for (int i = 0; i < HISTOGRAM_BINS; i++)
                table8Local(tempLocal[56](i) >> 14) = (uint8_t)(i);
            DataCopy(finaltable, table8Local, HISTOGRAM_BINS);
        }
    }
private:
    TPipe* pipe;
    TBuf<TPosition::VECCALC> compsize;
    TBuf<TPosition::VECCALC> compsizeprefix;
    TBuf<TPosition::VECCALC> table8;
    TBuf<TPosition::VECCALC> temp;
    TBuf<TPosition::VECCALC> datacopy;
    GlobalTensor<T> finalheader;
    GlobalTensor<uint8_t> finaltable;
    GlobalTensor<T> finalms;    
    GlobalTensor<T> finalmbl;
    GlobalTensor<T> finalcompprefix;
    GlobalTensor<uint8_t> compedexp;
    GlobalTensor<uint8_t> finalexp;
    GlobalTensor<T> histogram;
    GlobalTensor<T> blockcompsize;
    uint32_t blockId;
    uint32_t blockNum;
    uint32_t datablocksize;
    uint32_t datablocknum;
    uint32_t threadblocknum;
    uint32_t complevel;
    uint32_t totaluncompressedbytes;
    uint32_t totalcompressedbytes;
    uint32_t tilelength;
    uint32_t datatype;
    uint32_t mbllength;
    uint32_t options;
    uint32_t histogrambytes;
    uint32_t buffersize;
};
__global__ __aicore__ void merge(__gm__ uint8_t* finalHeader, __gm__ uint8_t* finalTable, __gm__ uint8_t* finalMs, __gm__ uint8_t* finalMbl, __gm__ uint8_t* finalCompPrefix, __gm__ uint8_t* compedexp, __gm__ uint8_t* finalExp, __gm__ uint8_t* histogramDevice, __gm__ uint8_t* blockCompSize, uint32_t dataBlockSize, uint32_t dataBlockNum, uint32_t threadBlockNum, uint32_t compLevel, uint32_t totalUncompressedBytes, uint32_t totalCompressedBytes, uint32_t tileLength, uint32_t dataType, uint32_t mblLength, uint32_t options, uint32_t histogramBytes, uint32_t bufferSize){
    TPipe pipe;
    mergeKernel<uint32_t> op;
    op.Init(&pipe, finalHeader, finalTable, finalMs, finalMbl, finalCompPrefix, compedexp, finalExp, histogramDevice, blockCompSize, dataBlockSize, dataBlockNum, threadBlockNum, compLevel, totalUncompressedBytes, totalCompressedBytes, tileLength, dataType, mblLength, options, histogramBytes, bufferSize);
    op.Process();
}
extern "C" void enec_merge(Header *cphd, void *stream, uint8_t *compressedDevice, uint8_t *compressedFinal, uint8_t *histogramDevice, uint8_t *blockCompSizeDevice, uint32_t bufferSize){
    merge<<<BLOCK_NUM, nullptr, stream>>>(compressedFinal, getTable(cphd, compressedFinal), getMsdata(cphd, compressedFinal), getMbl(cphd, compressedFinal), getCompSizePrefix(cphd, compressedFinal), getCompressed_exp(cphd, compressedDevice), getCompressed_exp(cphd, compressedFinal), histogramDevice, blockCompSizeDevice, cphd->dataBlockSize, cphd->dataBlockNum, cphd->threadBlockNum, cphd->compLevel, cphd->totalUncompressedBytes, cphd->totalCompressedBytes, cphd->tileLength, cphd->dataType, cphd->mblLength, cphd->options, cphd->histogramBytes, bufferSize);
}
