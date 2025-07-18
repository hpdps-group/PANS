/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
 * This file constains code of cpu debug and npu code.We read data from bin file
 * and write result to file.
 */

#include "snec_utils.h"
#include "snec_device.h"
template<typename T>
class Verify {
public:
    __aicore__ inline Verify() {}
    __aicore__ inline void Init(TPipe* pipe, uint32_t dataBlockNum, uint32_t dataBlockSize, uint32_t totalUndecompressedBytes, __gm__ uint8_t* decompressed, __gm__ uint8_t* src, __gm__ uint8_t* out) {
        this->pipe = pipe;
        this->dataBlockNum = dataBlockNum;
        this->blockId = GetBlockIdx();
        this->blockNum = GetBlockNum();
        this->dataBlockSize = dataBlockSize;
        this->totalUndecompressedBytes = totalUndecompressedBytes;
        decompressedGm.SetGlobalBuffer((__gm__ T*)(decompressed));
        srcGm.SetGlobalBuffer((__gm__ T*)(src));
        outGm.SetGlobalBuffer((__gm__ T*)(out));
        pipe->InitBuffer(decomp_inQueue, BUFFER_NUM, dataBlockSize);
        pipe->InitBuffer(src_inQueue, BUFFER_NUM, dataBlockSize);
        pipe->InitBuffer(outQueue, BUFFER_NUM, 32);
    }
public:
    __aicore__ inline void Process() {
        pipe->InitBuffer(temp, dataBlockSize);
        LocalTensor<uint8_t> tempLocal = temp.Get<uint8_t>();
        for(int i = blockId; i < dataBlockNum; i += blockNum){
            CopyIn(i);
            decompute(i, tempLocal);
            CopyOut(i);
        }
    }
    __aicore__ inline void CopyIn(int32_t i) {
        LocalTensor<T> decompLocal = decomp_inQueue.AllocTensor<T>();
        LocalTensor<T> srcLocal = src_inQueue.AllocTensor<T>();
        DataCopy(decompLocal, decompressedGm[i * dataBlockSize / sizeof(T)], dataBlockSize);
        DataCopy(srcLocal, srcGm[i * dataBlockSize / sizeof(T)], dataBlockSize);
        decomp_inQueue.EnQue(decompLocal);
        src_inQueue.EnQue(srcLocal);
    }
    __aicore__ inline void decompute(int32_t i, LocalTensor<uint8_t> tempLocal){
        LocalTensor<T> decompLocal = decomp_inQueue.DeQue<T>();
        LocalTensor<T> srcLocal = src_inQueue.DeQue<T>();
        LocalTensor<T> outLocal = outQueue.AllocTensor<T>();
        int num = 0;
        for(int i = 0; i < dataBlockSize / sizeof(T); i ++)
            if(decompLocal(i) != srcLocal(i)) 
                num = num + 1;
        outLocal(0) = num;
        if(blockId == 15){
            DumpTensor(decompLocal, 1, 256);
            DumpTensor(srcLocal, 1, 256);
            DumpTensor(outLocal, 1, 16);
        }
        outQueue.EnQue(outLocal);
        decomp_inQueue.FreeTensor(decompLocal);
        src_inQueue.FreeTensor(srcLocal);
    }
    __aicore__ inline void CopyOut(int32_t i){
        LocalTensor<T> outLocal = outQueue.DeQue<T>();
        DataCopy(outGm[i * 16], outLocal, 16);
        outQueue.FreeTensor(outLocal);
    }
private:
    TPipe* pipe;
    TQue<QuePosition::VECIN, 1> decomp_inQueue;
    TQue<QuePosition::VECOUT, 1> src_inQueue;
    TQue<QuePosition::VECOUT, 1> outQueue;
    TBuf<TPosition::VECCALC> temp;
    GlobalTensor<T> decompressedGm;
    GlobalTensor<T> srcGm;
    GlobalTensor<T> outGm;
    uint32_t blockId;
    uint32_t blockNum;
    uint32_t dataBlockNum;
    uint32_t dataBlockSize;
    uint32_t totalUndecompressedBytes;
};
__global__ __aicore__ void res_verify(uint32_t datablockNum, uint32_t datablockSize, uint32_t totalUndecompressedBytes, __gm__ uint8_t* decompressedDevice, __gm__ uint8_t* srcDevice, __gm__ uint8_t* outDevice) {
    TPipe pipe;
    Verify<int16_t> op;
    op.Init(&pipe, datablockNum, datablockSize, totalUndecompressedBytes, decompressedDevice, srcDevice, outDevice);
    op.Process();
}
extern "C" void enec_verify(Header *cphd, void *stream, uint8_t *decompressedDevice, uint8_t* srcDevice, uint8_t* outDevice){
    res_verify<<<cphd->threadBlockNum, nullptr, stream>>>(cphd->dataBlockNum, cphd->dataBlockSize, cphd->totalUncompressedBytes, decompressedDevice, srcDevice, outDevice);//计算前缀和，用于后续块合并，字节为单位，
}
