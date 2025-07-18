/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
 * This file constains code of cpu debug and npu code.We read data from bin file
 * and write result to file.
 */

#include "snec_utils.h"
#include "snec_device.h"
template <typename T>
class DecompressKernelBF16 {
public:
    __aicore__ inline DecompressKernelBF16() {}
    __aicore__ inline void Init(TPipe *pipe, uint32_t BUFFER_NUM, uint32_t elementNum, uint32_t tileLength, uint32_t tileNum, uint32_t threadblockNum, uint32_t datablockNum, uint32_t datablockSize, __gm__ uint8_t *eGlobal, __gm__ uint8_t *tableGlobal, __gm__ uint8_t *msGlobal, __gm__ uint8_t *mblGlobal, __gm__ uint8_t *compSizePrefix, __gm__ uint8_t *decompressedGlobal){
        this->pipe = pipe;
        this->blockId = GetBlockIdx();
        this->blockNum = GetBlockNum();
        this->computeNum = elementNum;
        this->tileLength = tileLength;
        this->tileNum = computeNum / tileLength;
        this->BLOCK_NUM = threadblockNum;
        this->datablockNum = datablockNum;
        this->datablockSize = datablockSize;
        srcShape_0[0] = tileNum;
        srcShape_0[1] = 1;
        dstShape_0[0] = tileNum;
        dstShape_0[1] = tileLength;
        srcShape_1[0] = 128;
        srcShape_1[1] = 1;
        dstShape_1[0] = 128;
        dstShape_1[1] = 64;
        table_input.SetGlobalBuffer((__gm__ T *)(tableGlobal));
        ms_input.SetGlobalBuffer((__gm__ T *)(msGlobal));
        mbl_input.SetGlobalBuffer((__gm__ T *)(mblGlobal));
        compSizePrefix_input.SetGlobalBuffer((__gm__ T *)(compSizePrefix));
        output.SetGlobalBuffer((__gm__ T *)(decompressedGlobal));
        pipe->InitBuffer(outQueue, BUFFER_NUM, computeNum * sizeof(T));
        pipe->InitBuffer(ms_inQueue, BUFFER_NUM, computeNum);
        pipe->InitBuffer(mbl_inQueue, BUFFER_NUM, tileNum * sizeof(T));
        pipe->InitBuffer(compPrefix, BLOCK_NUM * sizeof(T));
        LocalTensor<T> compPrefixLocal = compPrefix.Get<T>();
        AIV_WITH_BARRIER(DataCopy, compPrefixLocal, compSizePrefix_input, BLOCK_NUM);
        e_input.SetGlobalBuffer((__gm__ T *)(eGlobal + compPrefixLocal(blockId)));
    }

    __aicore__ inline void Process(){
        pipe->InitBuffer(e_in, computeNum * sizeof(T) + 32);
        pipe->InitBuffer(cmbl, tileNum * sizeof(T));
        pipe->InitBuffer(merge, computeNum * sizeof(T));
        pipe->InitBuffer(mblcmp, tileNum * sizeof(T));
        pipe->InitBuffer(take, tileNum * sizeof(T));
        pipe->InitBuffer(table, HISTOGRAM_BINS * sizeof(T));
        pipe->InitBuffer(table8, HISTOGRAM_BINS);
        pipe->InitBuffer(temp0, computeNum * sizeof(T));
        pipe->InitBuffer(temp1, computeNum * sizeof(T));
        pipe->InitBuffer(temp2, tileNum * sizeof(T));
        pipe->InitBuffer(temp3, tileNum * sizeof(T));
        pipe->InitBuffer(offset, 256 * sizeof(T));
        pipe->InitBuffer(mask15, tileNum * sizeof(T));
        LocalTensor<T> e_inLocal = e_in.Get<T>();
        LocalTensor<T> cmblLocal = cmbl.Get<T>();
        LocalTensor<T> mergeLocal = merge.Get<T>();
        LocalTensor<T> mblcmpLocal = mblcmp.Get<T>();
        LocalTensor<T> takeLocal = take.Get<T>();
        LocalTensor<T> tableLocal = table.Get<T>();
        LocalTensor<uint8_t> table8Local = table8.Get<uint8_t>();
        LocalTensor<T> tempLocal0 = temp0.Get<T>();
        LocalTensor<T> tempLocal1 = temp1.Get<T>();
        LocalTensor<T> tempLocal2 = temp2.Get<T>();
        LocalTensor<T> tempLocal3 = temp3.Get<T>();
        LocalTensor<T> offsetLocal = offset.Get<T>();
        LocalTensor<T> mask15Local = mask15.Get<T>();
        AIV_WITH_BARRIER(Duplicate, mask15Local, (T)15, tileNum);
        for (int i = 0; i < 256; i++){
            if (i % 2 == 0) offsetLocal(i) = (uint32_t)0;
            else offsetLocal(i) = (uint32_t)2147483648;
        }
        AIV_WITH_BARRIER(DataCopy, mergeLocal, e_input, computeNum / 2);
        int32_t eventIDMTE2ToV0 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV0);
        WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV0);
        AIV_WITH_BARRIER(ShiftRight, mergeLocal[computeNum / 2], mergeLocal, (uint32_t)16, computeNum / 2);
        AIV_WITH_BARRIER(ShiftLeft, tempLocal0, mergeLocal, (uint32_t)16, computeNum / 2);
        AIV_WITH_BARRIER(ShiftRight, mergeLocal, tempLocal0, (uint32_t)16, computeNum / 2);
        AIV_WITH_BARRIER(DataCopy, cmblLocal, mbl_input[blockId * tileNum / 8], tileNum / 8);
        int32_t eventIDMTE2ToV1 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV1);
        WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV1);
        AIV_WITH_BARRIER(ShiftRight, cmblLocal[tileNum / 8], cmblLocal, (uint32_t)16, tileNum / 8);
        AIV_WITH_BARRIER(ShiftRight, cmblLocal[tileNum / 4], cmblLocal, (uint32_t)8, tileNum / 4);
        AIV_WITH_BARRIER(ShiftRight, cmblLocal[tileNum / 2], cmblLocal, (uint32_t)4, tileNum / 2);
        AIV_WITH_BARRIER(And, cmblLocal, cmblLocal, mask15Local, tileNum * 2);
        for (int i = 0; i < 8; i++)
            takeLocal(i) = (((1 << i) - 1) << 14) | ((1 << i) << 5) | i;
        AIV_WITH_BARRIER(DataCopy, table8Local.template ReinterpretCast<T>(), table_input, HISTOGRAM_BINS / sizeof(T));
        int32_t eventIDMTE2ToV2 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV2);
        WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV2);
        for (int i = 0; i < HISTOGRAM_BINS; i++)
            tableLocal(i) = (uint32_t)table8Local(i);
        uint32_t accCompressed = computeNum * sizeof(uint16_t) / sizeof(T);
        int32_t remainderNum = datablockNum % blockNum;
        int32_t remainderStart = datablockNum - remainderNum;
        int32_t startdataBlock = blockId < remainderNum ? remainderStart + blockId : (remainderStart - blockNum) + blockId;
        for (int32_t i = startdataBlock; i > (int32_t)blockId; i -= (int32_t)blockNum){
            CopyIn_mbl(i);
            CopyIn_ms(i);
            Compute(i, accCompressed, e_inLocal, cmblLocal, mergeLocal, mblcmpLocal, takeLocal, tableLocal, tempLocal0, tempLocal1, tempLocal2, tempLocal3, offsetLocal, mask15Local);
            CopyOut(i);
            PipeBarrier<PIPE_ALL>();
        }
        CopyIn_ms(blockId);
        ComputeFirst(accCompressed, e_inLocal, cmblLocal, mergeLocal, mblcmpLocal, takeLocal, tableLocal, tempLocal0, tempLocal1, tempLocal2, tempLocal3, offsetLocal, mask15Local);
        CopyOut(blockId);
    }
private:
    __aicore__ inline void CopyIn_ms(uint32_t datablockId) {
        LocalTensor<T> ms_inLocal = ms_inQueue.AllocTensor<T>();
        DataCopy(ms_inLocal, ms_input[datablockId * (computeNum / 4)], computeNum / 4);
        ms_inQueue.EnQue(ms_inLocal);
    }
    __aicore__ inline void CopyIn_mbl(uint32_t datablockId) {
        LocalTensor<T> mbl_inLocal = mbl_inQueue.AllocTensor<T>();
        DataCopy(mbl_inLocal, mbl_input[datablockId * (tileNum / 8)], tileNum / 8);
        mbl_inQueue.EnQue(mbl_inLocal);
    }
    __aicore__ inline void Compute(int32_t i, uint32_t &accCompressed, LocalTensor<T> &e_inLocal, LocalTensor<T> &cmblLocal, LocalTensor<T> &mergeLocal, LocalTensor<T> &mblcmpLocal, LocalTensor<T> &takeLocal, LocalTensor<T> &tableLocal, LocalTensor<T> &tempLocal0, LocalTensor<T> &tempLocal1, LocalTensor<T> &tempLocal2, LocalTensor<T> &tempLocal3, LocalTensor<T> &offsetLocal, LocalTensor<T> &mask15Local) {
        LocalTensor<T> ms_inLocal = ms_inQueue.DeQue<T>();
        LocalTensor<T> mbl_inLocal = mbl_inQueue.DeQue<T>();
        LocalTensor<T> outLocal = outQueue.AllocTensor<T>();
        PipeBarrier<PIPE_ALL>();
        AIV_WITH_BARRIER(ShiftRight, mbl_inLocal[tileNum / 8], mbl_inLocal, (uint32_t)16, tileNum / 8);
        AIV_WITH_BARRIER(ShiftLeft, mbl_inLocal, mbl_inLocal, (uint32_t)16, tileNum / 4);
        AIV_WITH_BARRIER(ShiftRight, mbl_inLocal, mbl_inLocal, (uint32_t)16, tileNum / 4);
        AIV_WITH_BARRIER(ShiftRight, mbl_inLocal[tileNum / 4], mbl_inLocal, (uint32_t)8, tileNum / 4);
        AIV_WITH_BARRIER(ShiftLeft, mbl_inLocal, mbl_inLocal, (uint32_t)24, tileNum / 2);
        AIV_WITH_BARRIER(ShiftRight, mbl_inLocal, mbl_inLocal, (uint32_t)24, tileNum / 2);
        AIV_WITH_BARRIER(ShiftRight, mbl_inLocal[tileNum / 2], mbl_inLocal, (uint32_t)4, tileNum / 2);
        AIV_WITH_BARRIER(ShiftLeft, tempLocal0, mbl_inLocal, (uint32_t)28, tileNum / 2);
        AIV_WITH_BARRIER(ShiftRight, mbl_inLocal, tempLocal0, (uint32_t)28, tileNum / 2);
        AIV_WITH_BARRIER((Broadcast<float, 2, 1>), tempLocal0.template ReinterpretCast<float>(), cmblLocal.template ReinterpretCast<float>(), dstShape_0, srcShape_0);
        AIV_WITH_BARRIER(Adds, cmblLocal.template ReinterpretCast<int32_t>(), cmblLocal.template ReinterpretCast<int32_t>(), (int32_t)16, (int32_t)tileNum);
        AIV_WITH_BARRIER(Sub, cmblLocal.template ReinterpretCast<float>(), cmblLocal.template ReinterpretCast<float>(), mbl_inLocal.template ReinterpretCast<float>(), tileNum);
        AIV_WITH_BARRIER(ShiftLeft, cmblLocal, cmblLocal, (uint32_t)28, tileNum);
        AIV_WITH_BARRIER(ShiftRight, cmblLocal, cmblLocal, (uint32_t)28, tileNum);
        AIV_WITH_BARRIER(ShiftLeft, mbl_inLocal, mbl_inLocal, (uint32_t)2, tileNum);
        AIV_WITH_BARRIER(Gather, mbl_inLocal, takeLocal, mbl_inLocal, 0, tileNum);
        AIV_WITH_BARRIER((Broadcast<float, 2, 1>), tempLocal1.template ReinterpretCast<float>(), mbl_inLocal.template ReinterpretCast<float>(), dstShape_0, srcShape_0);
        AIV_WITH_BARRIER(ShiftLeft, outLocal, tempLocal1, (uint32_t)27, computeNum);
        AIV_WITH_BARRIER(ShiftRight, outLocal, outLocal, (uint32_t)27, computeNum);
        AIV_WITH_BARRIER(Compare, mblcmpLocal.template ReinterpretCast<uint8_t>(), tempLocal0.template ReinterpretCast<float>(), outLocal.template ReinterpretCast<float>(), CMPMODE::LT, computeNum);
        AIV_WITH_BARRIER(Duplicate, tempLocal0, (T)1, computeNum);
        AIV_WITH_BARRIER(Select, tempLocal0.template ReinterpretCast<float>(), mblcmpLocal, tempLocal0.template ReinterpretCast<float>(), (float)0, SELMODE::VSEL_TENSOR_SCALAR_MODE, computeNum);
        static constexpr CumSumConfig cumSumConfig{true, false, false};
        auto src0FLoat = tempLocal0.template ReinterpretCast<float>();
        auto dst0Float = outLocal.template ReinterpretCast<float>();
        auto lastRowFloat = mergeLocal.template ReinterpretCast<float>();
        auto sharedTmp = e_inLocal[8].template ReinterpretCast<uint8_t>();
        const CumSumInfo cumSumInfo0{128, 64};
        AIV_WITH_BARRIER((CumSum<float, cumSumConfig>), dst0Float, lastRowFloat, src0FLoat, sharedTmp, cumSumInfo0);
        uint64_t tempNum = 0;
        AIV_WITH_BARRIER(GatherMask, tempLocal2.template ReinterpretCast<float>(), outLocal.template ReinterpretCast<float>(), offsetLocal.template ReinterpretCast<uint32_t>(), true, computeNum, {1, 1, 1, 0}, tempNum);
        auto src1FLoat = tempLocal2.template ReinterpretCast<float>();
        auto dst1Float = tempLocal3.template ReinterpretCast<float>();
        const CumSumInfo cumSumInfo1{1, 128};
        AIV_WITH_BARRIER((CumSum<float, cumSumConfig>), dst1Float, lastRowFloat, src1FLoat, sharedTmp, cumSumInfo1);
        AIV_WITH_BARRIER((Broadcast<float, 2, 1>), tempLocal1.template ReinterpretCast<float>(), tempLocal3.template ReinterpretCast<float>(), dstShape_1, srcShape_1);
        AIV_WITH_BARRIER(Add, outLocal[64].template ReinterpretCast<int32_t>(), outLocal[64].template ReinterpretCast<int32_t>(), tempLocal1.template ReinterpretCast<int32_t>(), computeNum - 64);
        SCALAR_WITH_BARRIER(totalCompressed = outLocal(computeNum - 1) / 2);
        AIV_WITH_BARRIER(DataCopy, e_inLocal[8], e_input[accCompressed], totalCompressed);
        int32_t eventIDMTE2ToV = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
        SCALAR_WITH_BARRIER(accCompressed = accCompressed + totalCompressed);
        AIV_WITH_BARRIER(Adds, outLocal.template ReinterpretCast<int32_t>(), outLocal.template ReinterpretCast<int32_t>(), (int32_t)(-1), computeNum);
        AIV_WITH_BARRIER(ShiftLeft, outLocal, outLocal, (uint32_t)2, computeNum);
        AIV_WITH_BARRIER(Adds, outLocal.template ReinterpretCast<int32_t>(), outLocal.template ReinterpretCast<int32_t>(), (int32_t)32, computeNum);
        AIV_WITH_BARRIER(Mul, outLocal.template ReinterpretCast<int32_t>(), tempLocal0.template ReinterpretCast<int32_t>(), outLocal.template ReinterpretCast<int32_t>(), computeNum);
        AIV_WITH_BARRIER(ShiftRight, e_inLocal[8 + totalCompressed], e_inLocal[8], (uint32_t)16, totalCompressed);
        AIV_WITH_BARRIER(ShiftLeft, e_inLocal[8], e_inLocal[8], (uint32_t)16, totalCompressed);
        AIV_WITH_BARRIER(ShiftRight, e_inLocal[8], e_inLocal[8], (uint32_t)16, totalCompressed);
        AIV_WITH_BARRIER(Duplicate, e_inLocal, (T)0, 8);
        AIV_WITH_BARRIER(Gather, outLocal.template ReinterpretCast<float>(), e_inLocal.template ReinterpretCast<float>(), outLocal, (uint32_t)0, (uint32_t)computeNum);
        AIV_WITH_BARRIER(ShiftLeft, tempLocal0, mergeLocal, (uint32_t)16, computeNum);
        AIV_WITH_BARRIER(Or, tempLocal0.template ReinterpretCast<uint16_t>(), tempLocal0.template ReinterpretCast<uint16_t>(), outLocal.template ReinterpretCast<uint16_t>(), computeNum * 2);
        AIV_WITH_BARRIER(Select, mergeLocal.template ReinterpretCast<float>(), mblcmpLocal, tempLocal0.template ReinterpretCast<float>(), mergeLocal.template ReinterpretCast<float>(), SELMODE::VSEL_TENSOR_TENSOR_MODE, computeNum);
        AIV_WITH_BARRIER((Broadcast<float, 2, 1>), tempLocal1.template ReinterpretCast<float>(), mbl_inLocal.template ReinterpretCast<float>(), dstShape_0, srcShape_0);
        AIV_WITH_BARRIER(ShiftRight, tempLocal0, tempLocal1, (uint32_t)14, computeNum);
        AIV_WITH_BARRIER(And, outLocal.template ReinterpretCast<uint16_t>(), mergeLocal.template ReinterpretCast<uint16_t>(), tempLocal0.template ReinterpretCast<uint16_t>(), computeNum * 2);
        AIV_WITH_BARRIER(ShiftLeft, outLocal, outLocal, (uint32_t)2, computeNum);
        AIV_WITH_BARRIER(Gather, outLocal, tableLocal, outLocal, (uint32_t)0, (uint32_t)computeNum);
        AIV_WITH_BARRIER(ShiftLeft, tempLocal0, tempLocal1, (uint32_t)18, computeNum);
        AIV_WITH_BARRIER(ShiftRight, tempLocal0, tempLocal0, (uint32_t)23, computeNum);
        AIV_WITH_BARRIER(Div, mergeLocal.template ReinterpretCast<float>(), mergeLocal.template ReinterpretCast<float>(), tempLocal0.template ReinterpretCast<float>(), (int32_t)computeNum);
        AIV_WITH_BARRIER(Cast, mergeLocal.template ReinterpretCast<int32_t>(), mergeLocal.template ReinterpretCast<float>(), RoundMode::CAST_TRUNC, computeNum);
        AIV_WITH_BARRIER(ShiftRight, (tempLocal1.template ReinterpretCast<uint16_t>())[computeNum / 2], ms_inLocal.template ReinterpretCast<uint16_t>(), (uint16_t)8, computeNum / 2);
        AIV_WITH_BARRIER(ShiftLeft, tempLocal1.template ReinterpretCast<uint16_t>(), ms_inLocal.template ReinterpretCast<uint16_t>(), (uint16_t)8, computeNum / 2);
        AIV_WITH_BARRIER(ShiftRight, tempLocal1.template ReinterpretCast<uint16_t>(), tempLocal1.template ReinterpretCast<uint16_t>(), (uint16_t)8, computeNum / 2);
        AIV_WITH_BARRIER(ShiftLeft, outLocal, outLocal, (uint32_t)8, computeNum / 2);
        AIV_WITH_BARRIER(ShiftLeft, outLocal[computeNum / 2], outLocal[computeNum / 2], (uint32_t)24, computeNum / 2);
        AIV_WITH_BARRIER(Or, outLocal.template ReinterpretCast<uint16_t>(), outLocal.template ReinterpretCast<uint16_t>(), outLocal[computeNum / 2].template ReinterpretCast<uint16_t>(), computeNum / 2 * 2);
        AIV_WITH_BARRIER(Or, outLocal.template ReinterpretCast<uint16_t>(), outLocal.template ReinterpretCast<uint16_t>(), tempLocal1.template ReinterpretCast<uint16_t>(), computeNum * 2);
        AIV_WITH_BARRIER(ShiftLeft, tempLocal0.template ReinterpretCast<uint16_t>(), outLocal.template ReinterpretCast<uint16_t>(), (uint16_t)15, computeNum);
        AIV_WITH_BARRIER(ShiftRight, outLocal.template ReinterpretCast<uint16_t>(), outLocal.template ReinterpretCast<uint16_t>(), (uint16_t)1, computeNum);
        AIV_WITH_BARRIER(Or, outLocal.template ReinterpretCast<uint16_t>(), outLocal.template ReinterpretCast<uint16_t>(), tempLocal0.template ReinterpretCast<uint16_t>(), computeNum);
        outQueue.EnQue(outLocal);
        ms_inQueue.FreeTensor(ms_inLocal);
        mbl_inQueue.FreeTensor(mbl_inLocal);
    }
    __aicore__ inline void ComputeFirst(uint32_t &accCompressed, LocalTensor<T> &e_inLocal, LocalTensor<T> &cmblLocal, LocalTensor<T> &mergeLocal, LocalTensor<T> &mblcmpLocal, LocalTensor<T> &takeLocal, LocalTensor<T> &tableLocal, LocalTensor<T> &tempLocal0, LocalTensor<T> &tempLocal1, LocalTensor<T> &tempLocal2, LocalTensor<T> &tempLocal3, LocalTensor<T> &offsetLocal, LocalTensor<T> &mask15Local) {
        LocalTensor<T> ms_inLocal = ms_inQueue.DeQue<T>();
        LocalTensor<T> outLocal = outQueue.AllocTensor<T>();
        AIV_WITH_BARRIER(ShiftLeft, outLocal, mergeLocal, (uint32_t)2, computeNum);
        AIV_WITH_BARRIER(Gather, outLocal, tableLocal, outLocal, (uint32_t)0, (uint32_t)computeNum);
        AIV_WITH_BARRIER(ShiftRight, (tempLocal1.template ReinterpretCast<uint16_t>())[computeNum / 2], ms_inLocal.template ReinterpretCast<uint16_t>(), (uint16_t)8, computeNum / 2);
        AIV_WITH_BARRIER(ShiftLeft, tempLocal1.template ReinterpretCast<uint16_t>(), ms_inLocal.template ReinterpretCast<uint16_t>(), (uint16_t)8, computeNum / 2);
        AIV_WITH_BARRIER(ShiftRight, tempLocal1.template ReinterpretCast<uint16_t>(), tempLocal1.template ReinterpretCast<uint16_t>(), (uint16_t)8, computeNum / 2);
        AIV_WITH_BARRIER(ShiftLeft, outLocal, outLocal, (uint32_t)8, computeNum / 2);
        AIV_WITH_BARRIER(ShiftLeft, outLocal[computeNum / 2], outLocal[computeNum / 2], (uint32_t)24, computeNum / 2);
        AIV_WITH_BARRIER(Or, outLocal.template ReinterpretCast<uint16_t>(), outLocal.template ReinterpretCast<uint16_t>(), outLocal[computeNum / 2].template ReinterpretCast<uint16_t>(), computeNum / 2 * 2);
        AIV_WITH_BARRIER(Or, outLocal.template ReinterpretCast<uint16_t>(), outLocal.template ReinterpretCast<uint16_t>(), tempLocal1.template ReinterpretCast<uint16_t>(), computeNum * 2);
        AIV_WITH_BARRIER(ShiftLeft, tempLocal0.template ReinterpretCast<uint16_t>(), outLocal.template ReinterpretCast<uint16_t>(), (uint16_t)15, computeNum);
        AIV_WITH_BARRIER(ShiftRight, outLocal.template ReinterpretCast<uint16_t>(), outLocal.template ReinterpretCast<uint16_t>(), (uint16_t)1, computeNum);
        AIV_WITH_BARRIER(Or, outLocal.template ReinterpretCast<uint16_t>(), outLocal.template ReinterpretCast<uint16_t>(), tempLocal0.template ReinterpretCast<uint16_t>(), computeNum);
        outQueue.EnQue(outLocal);
        ms_inQueue.FreeTensor(ms_inLocal);
    }
    __aicore__ inline void CopyOut(uint32_t datablockId){
        LocalTensor<T> outLocal = outQueue.DeQue<T>();
        DataCopy(output[datablockId * (datablockSize / sizeof(T))], outLocal, datablockSize / sizeof(T));
        outQueue.FreeTensor(outLocal);
    }
private:
    TPipe *pipe;
    TQue<QuePosition::VECIN, 1> outQueue;
    TQue<QuePosition::VECOUT, 1> e_inQueue;
    TQue<QuePosition::VECOUT, 1> ms_inQueue;
    TQue<QuePosition::VECOUT, 1> mbl_inQueue;
    TBuf<TPosition::VECCALC> compPrefix;
    TBuf<TPosition::VECCALC> e_in;
    TBuf<TPosition::VECCALC> cmbl;
    TBuf<TPosition::VECCALC> merge;
    TBuf<TPosition::VECCALC> mblcmp;
    TBuf<TPosition::VECCALC> mblcmp32;
    TBuf<TPosition::VECCALC> take;
    TBuf<TPosition::VECCALC> table;
    TBuf<TPosition::VECCALC> table8;
    TBuf<TPosition::VECCALC> temp0;
    TBuf<TPosition::VECCALC> temp1;
    TBuf<TPosition::VECCALC> temp2;
    TBuf<TPosition::VECCALC> temp3;
    TBuf<TPosition::VECCALC> offset;
    TBuf<TPosition::VECCALC> mask15;
    GlobalTensor<T> e_input;
    GlobalTensor<T> table_input;
    GlobalTensor<T> ms_input;
    GlobalTensor<T> mbl_input;
    GlobalTensor<T> output;
    GlobalTensor<T> compSizePrefix_input;
    uint32_t blockId;
    uint32_t blockNum;
    uint32_t computeNum;
    uint32_t tileLength;
    uint32_t tileNum;
    uint32_t BLOCK_NUM;
    uint32_t datablockNum;
    uint32_t datablockSize;
    uint32_t totalCompressed;
    uint32_t srcShape_0[2];
    uint32_t dstShape_0[2];
    uint32_t srcShape_1[2];
    uint32_t dstShape_1[2];
};
__global__ __aicore__ void decompBF16(uint32_t BUFFER_NUM, uint32_t elementNum, uint32_t tileLength, uint32_t tileNum, uint32_t threadblockNum, uint32_t datablockNum, uint32_t datablockSize, __gm__ uint8_t* eGlobal, __gm__ uint8_t* tableGlobal, __gm__ uint8_t* msGlobal, __gm__ uint8_t* mblGlobal, __gm__ uint8_t* compSizePrefix, __gm__ uint8_t* decompressedGlobal){
    TPipe pipe;
    DecompressKernelBF16<uint32_t> op;
    op.Init(&pipe, BUFFER_NUM, elementNum, tileLength, tileNum, threadblockNum, datablockNum, datablockSize, eGlobal, tableGlobal, msGlobal, mblGlobal, compSizePrefix, decompressedGlobal);
    op.Process();
}
extern "C" void enec_decompress(Header* cphd, void* stream, uint8_t* compressed, uint8_t* decompressed) {
    switch (cphd->dataType){
    case 0:{ // BF16
        decompBF16<<<cphd->threadBlockNum, nullptr, stream>>>(1, cphd->dataBlockSize / sizeof(uint16_t), cphd->tileLength, elementNum / cphd->tileLength, cphd->threadBlockNum, cphd->dataBlockNum, cphd->dataBlockSize, getCompressed_exp(cphd, compressed), getTable(cphd, compressed), getMsdata(cphd, compressed), getMbl(cphd, compressed), getCompSizePrefix(cphd, compressed), decompressed);
        break;
    }
    case 1:{ // FP16

        break;
    }
    case 2:{ // FP32

        break;
    }
    default:{

        return;
    }
    }
}
