/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
 * This file constains code of cpu debug and npu code.We read data from bin file
 * and write result to file.
 */

#include "snec_utils.h"
#include "snec_device.h"
template <typename T>
class CompressKernelBF16 {
public:
    __aicore__ inline CompressKernelBF16() {}
    __aicore__ inline void Init(TPipe *pipe, uint32_t datablockNum, uint32_t datablockSize, uint32_t elementNum, uint32_t tileLength, __gm__ uint8_t *srcDevice, __gm__ uint8_t *msGlobal, __gm__ uint8_t *mblGlobal, __gm__ uint8_t *eGlobal, __gm__ uint8_t *histogramDevice, __gm__ uint8_t *blockCompSize) {
        this->pipe = pipe;
        this->blockId = GetBlockIdx();
        this->blockNum = GetBlockNum();
        this->computeNum = elementNum;
        this->tileLength = tileLength;
        this->tileNum = elementNum / tileLength;
        this->datablockNum = datablockNum;
        this->datablockSize = datablockSize;
        int datablockNumPerBLOCK = (datablockNum + blockNum - 1) / blockNum;
        this->bufferSize = (datablockSize / 2 * datablockNumPerBLOCK);
        srcShape_[0] = tileNum;
        srcShape_[1] = 1;
        dstShape_[0] = tileNum;
        dstShape_[1] = tileLength;
        input.SetGlobalBuffer((__gm__ T *)(srcDevice));
        table_input.SetGlobalBuffer((__gm__ T *)(histogramDevice));
        ms_output.SetGlobalBuffer((__gm__ T *)(msGlobal));
        mbl_output.SetGlobalBuffer((__gm__ T *)(mblGlobal));
        e_output.SetGlobalBuffer((__gm__ T *)(eGlobal + bufferSize * blockId));
        blockCompSizeOutput.SetGlobalBuffer((__gm__ uint32_t *)(blockCompSize + 32 * blockId));
        pipe->InitBuffer(inQueue, BUFFER_NUM, computeNum * sizeof(T));
        pipe->InitBuffer(e_outQueue, BUFFER_NUM, computeNum * sizeof(T));
        pipe->InitBuffer(ms_outQueue, BUFFER_NUM, computeNum);
        pipe->InitBuffer(mbl_outQueue, BUFFER_NUM, tileNum);
    }
    __aicore__ inline void Process(){
        pipe->InitBuffer(temp0, computeNum * sizeof(T));
        pipe->InitBuffer(table, HISTOGRAM_BINS * sizeof(T));
        pipe->InitBuffer(merge, computeNum * sizeof(T));
        pipe->InitBuffer(cmbl, tileNum * sizeof(T));
        pipe->InitBuffer(mask15, tileNum * sizeof(T));
        pipe->InitBuffer(mask16383, tileNum * sizeof(T));
        LocalTensor<T> tempLocal0 = temp0.Get<T>();
        LocalTensor<T> tableLocal = table.Get<T>();
        LocalTensor<T> mergeLocal = merge.Get<T>();
        LocalTensor<T> cmblLocal = cmbl.Get<T>();
        LocalTensor<T> mask15Local = mask15.Get<T>();
        LocalTensor<T> mask16383Local = mask16383.Get<T>();
        AIV_WITH_BARRIER(DataCopy, tableLocal, table_input, HISTOGRAM_BINS);
        AIV_WITH_BARRIER(Duplicate, tempLocal0, (T)0, computeNum);
        AIV_WITH_BARRIER(Duplicate, mergeLocal, (T)0, computeNum);
        AIV_WITH_BARRIER(Duplicate, cmblLocal, (T)0, tileNum);
        AIV_WITH_BARRIER(Duplicate, mask15Local, (T)15, tileNum);
        AIV_WITH_BARRIER(Duplicate, mask16383Local, (T)16383, tileNum);
        uint64_t compressedSize = 0;
        uint32_t totalcompressedSize = 0;
        for (uint32_t i = blockId; i < datablockNum; i += blockNum){
            uint32_t offset = i * (computeNum * sizeof(uint16_t) / sizeof(T));
            CopyIn(offset);
            Compute(i, compressedSize, tempLocal0, tableLocal, mergeLocal, cmblLocal, mask15Local, mask16383Local);
            compressedSize = compressedSize * sizeof(uint16_t);
            totalcompressedSize = totalcompressedSize + (uint32_t)compressedSize;
            CopyOut(totalcompressedSize, compressedSize, i);
        }
        AIV_WITH_BARRIER(ShiftLeft, cmblLocal[tileNum / 2], cmblLocal[tileNum / 2], (uint32_t)4, tileNum / 2);
        AIV_WITH_BARRIER(Or, cmblLocal, cmblLocal, cmblLocal[tileNum / 2], tileNum / 2 * 2);
        AIV_WITH_BARRIER(ShiftLeft, cmblLocal[tileNum / 4], cmblLocal[tileNum / 4], (uint32_t)8, tileNum / 4);
        AIV_WITH_BARRIER(Or, cmblLocal, cmblLocal, cmblLocal[tileNum / 4], tileNum / 4 * 2);
        AIV_WITH_BARRIER(ShiftLeft, cmblLocal[tileNum / 8], cmblLocal[tileNum / 8], (uint32_t)16, tileNum / 8);
        AIV_WITH_BARRIER(Or, cmblLocal, cmblLocal, cmblLocal[tileNum / 8], tileNum / 8 * 2);
        AIV_WITH_BARRIER(DataCopy, mbl_output[blockId * tileNum / 8], cmblLocal, tileNum / 8);
        AIV_WITH_BARRIER(ShiftLeft, mergeLocal[computeNum / 2], mergeLocal[computeNum / 2], (uint32_t)16, computeNum / 2);
        AIV_WITH_BARRIER(Or, mergeLocal, mergeLocal, mergeLocal[computeNum / 2], computeNum / 2 * 2);
        totalcompressedSize = totalcompressedSize + computeNum * 2;
        AIV_WITH_BARRIER(DataCopy, e_output[(bufferSize - totalcompressedSize) / sizeof(T)], mergeLocal, computeNum / 2);
        tempLocal0(0) = totalcompressedSize;
        AIV_WITH_BARRIER(DataCopy, blockCompSizeOutput, tempLocal0, 32 / sizeof(T));
    }
private:
    __aicore__ inline void CopyIn(uint32_t offset) {
        LocalTensor<T> inLocal = inQueue.AllocTensor<T>();
        AIV_WITHOUT_BARRIER(DataCopy, inLocal, input[offset], computeNum * sizeof(uint16_t) / sizeof(T));
        inQueue.EnQue(inLocal);
    }
    __aicore__ inline void Compute(uint32_t i, uint64_t &compressedSize, LocalTensor<T> &tempLocal0, LocalTensor<T> &tableLocal, LocalTensor<T> &mergeLocal, LocalTensor<T> &cmblLocal, LocalTensor<T> &mask15Local, LocalTensor<T> &mask16383Local) {
        LocalTensor<T> e_inLocal = inQueue.DeQue<T>();
        LocalTensor<T> e_outLocal = e_outQueue.AllocTensor<T>();
        LocalTensor<T> mbl_outLocal = mbl_outQueue.AllocTensor<T>();
        LocalTensor<T> ms_outLocal = ms_outQueue.AllocTensor<T>();
        AIV_WITH_BARRIER(ShiftRight, tempLocal0.template ReinterpretCast<uint16_t>(), e_inLocal.template ReinterpretCast<uint16_t>(), (uint16_t)15, computeNum);
        AIV_WITH_BARRIER(ShiftLeft, e_inLocal.template ReinterpretCast<uint16_t>(), e_inLocal.template ReinterpretCast<uint16_t>(), (uint16_t)1, computeNum);
        AIV_WITH_BARRIER(Or, e_inLocal.template ReinterpretCast<uint16_t>(), e_inLocal.template ReinterpretCast<uint16_t>(), tempLocal0.template ReinterpretCast<uint16_t>(), computeNum);
        AIV_WITH_BARRIER(ShiftLeft, tempLocal0.template ReinterpretCast<uint16_t>(), e_inLocal.template ReinterpretCast<uint16_t>(), (uint16_t)8, computeNum);
        AIV_WITH_BARRIER(ShiftRight, tempLocal0.template ReinterpretCast<uint16_t>(), tempLocal0.template ReinterpretCast<uint16_t>(), (uint16_t)8, (int)(computeNum / 2));
        AIV_WITH_BARRIER(Or, ms_outLocal.template ReinterpretCast<uint16_t>(), tempLocal0.template ReinterpretCast<uint16_t>(), (tempLocal0.template ReinterpretCast<uint16_t>())[computeNum / 2], computeNum / 2);
        AIV_WITH_BARRIER(ShiftRight, e_inLocal.template ReinterpretCast<uint16_t>(), e_inLocal.template ReinterpretCast<uint16_t>(), (uint16_t)8, computeNum);
        AIV_WITH_BARRIER(ShiftRight, e_inLocal[computeNum / 2], e_inLocal, (uint32_t)16, computeNum / 2);
        AIV_WITH_BARRIER(ShiftLeft, e_inLocal, e_inLocal, (uint32_t)16, computeNum / 2);
        AIV_WITH_BARRIER(ShiftRight, e_inLocal, e_inLocal, (uint32_t)16, computeNum / 2);
        AIV_WITH_BARRIER(ShiftLeft, e_inLocal, e_inLocal, (uint32_t)2, computeNum);
        AIV_WITH_BARRIER(Gather, e_inLocal, tableLocal, e_inLocal, (uint32_t)0, (uint32_t)computeNum);
        AIV_WITH_BARRIER(ShiftRight, e_outLocal, e_inLocal, (uint32_t)14, computeNum);
        AIV_WITH_BARRIER(WholeReduceMax<float>, (e_inLocal.template ReinterpretCast<float>())[0], (e_inLocal.template ReinterpretCast<float>())[0 * tileLength * 128], tileLength, 128, 1, 1, tileLength * sizeof(T) / 32, ReduceOrder::ORDER_ONLY_VALUE);
        AIV_WITH_BARRIER(WholeReduceMax<float>, (e_inLocal.template ReinterpretCast<float>())[128], (e_inLocal.template ReinterpretCast<float>())[1 * tileLength * 128], tileLength, 128, 1, 1, tileLength * sizeof(T) / 32, ReduceOrder::ORDER_ONLY_VALUE);
        AIV_WITH_BARRIER(WholeReduceMax<float>, (e_inLocal.template ReinterpretCast<float>())[256], (e_inLocal.template ReinterpretCast<float>())[2 * tileLength * 128], tileLength, 128, 1, 1, tileLength * sizeof(T) / 32, ReduceOrder::ORDER_ONLY_VALUE);
        AIV_WITH_BARRIER(WholeReduceMax<float>, (e_inLocal.template ReinterpretCast<float>())[384], (e_inLocal.template ReinterpretCast<float>())[3 * tileLength * 128], tileLength, 128, 1, 1, tileLength * sizeof(T) / 32, ReduceOrder::ORDER_ONLY_VALUE);
        AIV_WITH_BARRIER(And, e_inLocal, e_inLocal, mask16383Local, tileNum * 2);
        AIV_WITH_BARRIER(Add, cmblLocal.template ReinterpretCast<int32_t>(), cmblLocal.template ReinterpretCast<int32_t>(), e_inLocal.template ReinterpretCast<int32_t>(), tileNum);
        AIV_WITH_BARRIER(And, mbl_outLocal, e_inLocal, mask15Local, tileNum * 2);
        AIV_WITH_BARRIER(ShiftLeft, mbl_outLocal[tileNum / 2], mbl_outLocal[tileNum / 2], (uint32_t)4, tileNum / 2);
        AIV_WITH_BARRIER(Or, mbl_outLocal, mbl_outLocal, mbl_outLocal[tileNum / 2], tileNum / 2 * 2);
        AIV_WITH_BARRIER(ShiftLeft, mbl_outLocal[tileNum / 4], mbl_outLocal[tileNum / 4], (uint32_t)8, tileNum / 4);
        AIV_WITH_BARRIER(Or, mbl_outLocal, mbl_outLocal, mbl_outLocal[tileNum / 4], tileNum / 4 * 2);
        AIV_WITH_BARRIER(ShiftLeft, mbl_outLocal[tileNum / 8], mbl_outLocal[tileNum / 8], (uint32_t)16, tileNum / 8);
        AIV_WITH_BARRIER(Or, mbl_outLocal, mbl_outLocal, mbl_outLocal[tileNum / 8], tileNum / 8 * 2);
        AIV_WITH_BARRIER((Broadcast<float, 2, 1>), e_inLocal.template ReinterpretCast<float>(), cmblLocal.template ReinterpretCast<float>(), dstShape_, srcShape_);
        AIV_WITH_BARRIER(And, cmblLocal, cmblLocal, mask15Local, tileNum * 2);
        AIV_WITH_BARRIER(ShiftLeft, tempLocal0, e_inLocal, (uint32_t)27, computeNum);
        AIV_WITH_BARRIER(ShiftRight, tempLocal0, tempLocal0, (uint32_t)27, computeNum);
        AIV_WITH_BARRIER(CompareScalar, tempLocal0.template ReinterpretCast<uint8_t>(), tempLocal0.template ReinterpretCast<float>(), (mask15Local.template ReinterpretCast<float>())(0), CMPMODE::GT, computeNum);
        AIV_WITH_BARRIER(ShiftRight, e_inLocal, e_inLocal, (uint32_t)5, computeNum);
        AIV_WITH_BARRIER(Mul, mergeLocal.template ReinterpretCast<int32_t>(), mergeLocal.template ReinterpretCast<int32_t>(), e_inLocal.template ReinterpretCast<int32_t>(), (int32_t)computeNum);
        AIV_WITH_BARRIER(Or, mergeLocal, mergeLocal, e_outLocal, (int32_t)computeNum * 2);
        AIV_WITH_BARRIER(GatherMask, e_outLocal.template ReinterpretCast<float>(), mergeLocal.template ReinterpretCast<float>(), tempLocal0.template ReinterpretCast<uint32_t>(), true, computeNum, {1, 1, 1, 0}, compressedSize);
        AIV_WITH_BARRIER(ShiftLeft, e_outLocal, e_outLocal, (uint32_t)16, compressedSize);
        AIV_WITH_BARRIER(ShiftRight, e_outLocal, e_outLocal, (uint32_t)16, compressedSize / 2);
        AIV_WITH_BARRIER(Or, e_outLocal, e_outLocal, e_outLocal[compressedSize / 2], compressedSize / 2 * 2);
        AIV_WITH_BARRIER(ShiftRight, e_inLocal, mergeLocal, (uint32_t)16, computeNum);
        AIV_WITH_BARRIER(Select, mergeLocal.template ReinterpretCast<float>(), tempLocal0, e_inLocal.template ReinterpretCast<float>(), mergeLocal.template ReinterpretCast<float>(), SELMODE::VSEL_TENSOR_TENSOR_MODE, computeNum);
        e_outQueue.EnQue(e_outLocal);
        ms_outQueue.EnQue(ms_outLocal);
        mbl_outQueue.EnQue(mbl_outLocal);
        inQueue.FreeTensor(e_inLocal);
    }
    __aicore__ inline void CopyOut(uint32_t totalcompressedSize, uint32_t compressedSize, uint32_t datablockId) {
        LocalTensor<T> e_outLocal = e_outQueue.DeQue<T>();
        LocalTensor<T> mbl_outLocal = mbl_outQueue.DeQue<T>();
        LocalTensor<T> ms_outLocal = ms_outQueue.DeQue<T>();
        AIV_WITHOUT_BARRIER(DataCopy, e_output[(bufferSize - totalcompressedSize) / sizeof(T)], e_outLocal, compressedSize / sizeof(T));
        AIV_WITHOUT_BARRIER(DataCopy, ms_output[datablockId * ((computeNum / 2) * sizeof(uint16_t) / sizeof(T))], ms_outLocal, ((computeNum / 2) * sizeof(uint16_t) / sizeof(T)));
        AIV_WITHOUT_BARRIER(DataCopy, mbl_output[datablockId * tileNum / 8], mbl_outLocal, tileNum / 8);
        e_outQueue.FreeTensor(e_outLocal);
        ms_outQueue.FreeTensor(ms_outLocal);
        mbl_outQueue.FreeTensor(mbl_outLocal);
    }
private:
    TPipe *pipe;
    TQue<QuePosition::VECIN, 1> inQueue;
    TQue<QuePosition::VECOUT, 1> e_outQueue;
    TQue<QuePosition::VECOUT, 1> ms_outQueue;
    TQue<QuePosition::VECOUT, 1> mbl_outQueue;
    TBuf<TPosition::VECCALC> temp0;
    TBuf<TPosition::VECCALC> table;
    TBuf<TPosition::VECCALC> merge;
    TBuf<TPosition::VECCALC> cmbl;
    TBuf<TPosition::VECCALC> mask15;
    TBuf<TPosition::VECCALC> mask16383;
    GlobalTensor<T> input;
    GlobalTensor<T> table_input;
    GlobalTensor<T> mbl_output;
    GlobalTensor<T> e_output;
    GlobalTensor<T> ms_output;
    GlobalTensor<T> blockCompSizeOutput;
    uint32_t blockId;
    uint32_t blockNum;
    uint32_t computeNum;
    uint32_t tileLength;
    uint32_t tileNum;
    uint32_t threadblockNum;
    uint32_t datablockNum;
    uint32_t datablockSize;
    uint32_t bufferSize;
    uint32_t srcShape_[2];
    uint32_t dstShape_[2];
};
__global__ __aicore__ void compBF16(uint32_t datablockNum, uint32_t datablockSize, uint32_t elementNum, uint32_t tileLength, __gm__ uint8_t* srcDevice, __gm__ uint8_t* msGlobal, __gm__ uint8_t* mblGlobal, __gm__ uint8_t* eGlobal, __gm__ uint8_t* histogramDevice, __gm__ uint8_t* blockCompSize) {
    TPipe pipe;
    CompressKernelBF16<uint32_t> op;
    op.Init(&pipe, datablockNum, datablockSize, elementNum, tileLength, srcDevice, msGlobal, mblGlobal, eGlobal, histogramDevice, blockCompSize);
    op.Process();
}
extern "C" void enec_compress(Header *cphd, void *stream, uint8_t* srcDevice, uint8_t* compressedDevice, uint8_t* compressedFinal, uint8_t* histogramDevice, uint8_t* blockCompSize)
{
    switch (cphd->dataType){
    case 0:{ // BF16
        uint32_t elementNum = cphd->dataBlockSize / sizeof(uint16_t);
        compBF16<<<BLOCK_NUM, nullptr, stream>>>(cphd->dataBlockNum, cphd->dataBlockSize, elementNum, cphd->tileLength, srcDevice, getMsdata(cphd, compressedFinal), getMbl(cphd, compressedFinal), getCompressed_exp(cphd, compressedDevice), histogramDevice, blockCompSize);
        break;
    }
    case 1:{ // FP16

        break;
    }
    case 2:{ // FP32

        break;
    }
    default:
        return;
    }
}
