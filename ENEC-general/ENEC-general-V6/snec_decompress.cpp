/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
 * This file constains code of cpu debug and npu code.We read data from bin file
 * and write result to file.
 */

#include "snec_utils.h"
#include "snec_device.h"

template <typename T>
class DecompressKernelBF16
{
public:
    __aicore__ inline DecompressKernelBF16() {}

    __aicore__ inline void Init(TPipe *pipe,
                                uint32_t BUFFER_NUM,
                                uint32_t elementNum,
                                uint32_t tileLength,
                                uint32_t tileNum,
                                uint32_t threadblockNum,
                                uint32_t datablockNum,
                                uint32_t datablockSize,
                                __gm__ uint8_t *eGlobal,           // e_input
                                __gm__ uint8_t *tableGlobal,       // table_input
                                __gm__ uint8_t *msGlobal,          // ms_input
                                __gm__ uint8_t *mblGlobal,         // mbl_input
                                __gm__ uint8_t *compSizePrefix,    // compSizePrefix
                                __gm__ uint8_t *decompressedGlobal // output
    )
    {
        this->pipe = pipe;
        this->blockId = GetBlockIdx();
        this->blockNum = GetBlockNum();
        this->computeNum = elementNum;
        this->tileLength = tileLength;
        this->tileNum = computeNum / tileLength;
        this->BLOCK_NUM = threadblockNum;
        this->datablockNum = datablockNum;
        this->datablockSize = datablockSize;

        // srcShape_0[0] = tileNum;
        // srcShape_0[1] = 1;
        // dstShape_0[0] = tileNum;
        // dstShape_0[1] = tileLength;

        srcShape_1[0] = 64;
        srcShape_1[1] = 1;
        dstShape_1[0] = 64;
        dstShape_1[1] = 8;
        // srcShape_1[0] = 32;
        // srcShape_1[1] = 1;
        // dstShape_1[0] = 32;
        // dstShape_1[1] = 16;

        srcShape_prefix[0] = 1;
        srcShape_prefix[1] = tileNum;
        dstShape_prefix[0] = tileLength;
        dstShape_prefix[1] = tileNum;

        srcShape_offset[0] = tileLength;
        srcShape_offset[1] = 1;
        dstShape_offset[0] = tileLength;
        dstShape_offset[1] = tileNum;

        srcShape_mblcmp[0] = 1;
        srcShape_mblcmp[1] = tileNum / 8 / sizeof(T);
        dstShape_mblcmp[0] = tileLength;
        dstShape_mblcmp[1] = tileNum / 8 / sizeof(T);

        srcShape_div_fetch[0] = 1;
        srcShape_div_fetch[1] = tileNum;
        dstShape_div_fetch[0] = tileLength;
        dstShape_div_fetch[1] = tileNum;

        table_input.SetGlobalBuffer((__gm__ T *)(tableGlobal));
        ms_input.SetGlobalBuffer((__gm__ T *)(msGlobal));
        mbl_input.SetGlobalBuffer((__gm__ T *)(mblGlobal));
        compSizePrefix_input.SetGlobalBuffer((__gm__ T *)(compSizePrefix));
        output.SetGlobalBuffer((__gm__ T *)(decompressedGlobal));

        pipe->InitBuffer(outQueue, BUFFER_NUM, computeNum * sizeof(T));// 32kb
        pipe->InitBuffer(ms_inQueue, BUFFER_NUM, computeNum 
            //* sizeof(uint16_t)
            );// 8kb
        pipe->InitBuffer(mbl_inQueue, BUFFER_NUM, tileNum * sizeof(T));// 2kb
        pipe->InitBuffer(compPrefix, BLOCK_NUM * sizeof(T));// 192b

        LocalTensor<T> compPrefixLocal = compPrefix.Get<T>();
        AIV_WITH_BARRIER(DataCopy, compPrefixLocal, compSizePrefix_input, BLOCK_NUM);
        e_input.SetGlobalBuffer((__gm__ T *)(eGlobal + compPrefixLocal(blockId)));
    }

    __aicore__ inline void Process()
    {
        pipe->InitBuffer(e_in, computeNum * sizeof(T) + 32);
        pipe->InitBuffer(cmbl, tileNum * sizeof(T));
        pipe->InitBuffer(merge, computeNum * sizeof(T));
        pipe->InitBuffer(mblcmp, computeNum / 8);
        pipe->InitBuffer(take, 32 * sizeof(T));
        pipe->InitBuffer(table, HISTOGRAM_BINS * sizeof(T));
        pipe->InitBuffer(table8, HISTOGRAM_BINS);
        pipe->InitBuffer(temp0, computeNum * sizeof(T));
        // pipe->InitBuffer(temp1, computeNum * sizeof(T));
        pipe->InitBuffer(temp2, tileNum * sizeof(T));
        // pipe->InitBuffer(temp3, tileNum * sizeof(T));
        pipe->InitBuffer(offset0, tileLength * sizeof(T));
        pipe->InitBuffer(offset1, tileLength * sizeof(T));
        pipe->InitBuffer(div_fetch, tileNum * sizeof(T));
        pipe->InitBuffer(mask1, tileNum * sizeof(T));
        pipe->InitBuffer(mask15, tileNum * sizeof(T));
        pipe->InitBuffer(mask257, tileNum * sizeof(float));
        pipe->InitBuffer(maskne1, tileNum * sizeof(float));

        LocalTensor<T> e_inLocal = e_in.Get<T>();
        LocalTensor<T> cmblLocal = cmbl.Get<T>();
        LocalTensor<T> mergeLocal = merge.Get<T>();
        LocalTensor<T> mblcmpLocal = mblcmp.Get<T>();
        LocalTensor<T> takeLocal = take.Get<T>();
        LocalTensor<T> tableLocal = table.Get<T>();
        LocalTensor<uint8_t> table8Local = table8.Get<uint8_t>();
        LocalTensor<T> tempLocal0 = temp0.Get<T>();
        // LocalTensor<T> tempLocal1 = temp1.Get<T>();
        LocalTensor<T> tempLocal2 = temp2.Get<T>();
        // LocalTensor<T> tempLocal3 = temp3.Get<T>();
        LocalTensor<T> offset0Local = offset0.Get<T>();
        LocalTensor<T> offset1Local = offset1.Get<T>();
        LocalTensor<T> tempLocal3 = div_fetch.Get<T>();
        LocalTensor<T> mask1Local = mask1.Get<T>();
        LocalTensor<T> mask15Local = mask15.Get<T>();
        LocalTensor<float> mask257Local = mask257.Get<float>();
        LocalTensor<float> maskne1Local = maskne1.Get<float>();
        
        for(int i = 0; i < tileLength; i++){
            offset0Local(i) = 
            // 2147516416
            2155905152
            ;
        }

        AIV_WITH_BARRIER(CreateVecIndex, offset1Local.template ReinterpretCast<int32_t>(), 0, tileLength);

        AIV_WITH_BARRIER(Duplicate, mask1Local, (T)1, tileNum);
        AIV_WITH_BARRIER(Duplicate, mask15Local, (T)15, tileNum);
        AIV_WITH_BARRIER(Duplicate, mask257Local, (float)257, tileNum);
        AIV_WITH_BARRIER(Duplicate, maskne1Local, (float)-1, tileNum);

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

        for (int i = 0; i <= 8; i++)
        {
            uint32_t extra = (1 << i) - 1;
            uint32_t divNum = 1 << i;
            uint32_t mbl = i;
            takeLocal(i) = (extra << 14) | (divNum << 5) | (mbl);
        }

        AIV_WITH_BARRIER(DataCopy, table8Local.template ReinterpretCast<T>(), table_input, HISTOGRAM_BINS / sizeof(T));
        int32_t eventIDMTE2ToV2 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV2);
        WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV2);
        for (int i = 0; i < HISTOGRAM_BINS; i++)
        {
            tableLocal(i) = (uint32_t)table8Local(i);
        }

        uint32_t accCompressed = computeNum * sizeof(uint16_t) / sizeof(T);
        uint64_t tempNum = 0;
        int32_t remainderNum = datablockNum % blockNum;
        int32_t remainderStart = datablockNum - remainderNum;
        int32_t startdataBlock = blockId < remainderNum ? remainderStart + blockId : (remainderStart - blockNum) + blockId;

        auto
        // LocalTensor<float>& 
        src1Float = tempLocal2.template ReinterpretCast<float>();
        auto
        // LocalTensor<float>& 
        dst1Float = tempLocal3.template ReinterpretCast<float>();

        for (int32_t i = startdataBlock; i >
            // startdataBlock - (int32_t)blockNum * 1
            (int32_t)blockId
            ; i -= (int32_t)blockNum)
        {
            // if(
            //     //blockId == 20
            //     i == datablockNum - 1 - blockNum 
            //     || 
            //     i == datablockNum - 1
            //     ){
            CopyIn_mbl(i);
            CopyIn_ms(i);
            Compute(
                i,
                tempNum,
                accCompressed,
                e_inLocal,
                cmblLocal,
                mergeLocal,
                mblcmpLocal,
                takeLocal,
                tableLocal,
                tempLocal0,
                tempLocal2,
                offset0Local,
                offset1Local,
                tempLocal3, 
                mask1Local, 
                mask15Local,
                mask257Local,
                maskne1Local,
                src1Float,
                dst1Float
            );
            CopyOut(i);
            PipeBarrier<PIPE_ALL>();
            // }
        }
        // CopyIn_ms(blockId);
        // ComputeFirst(
        //     accCompressed,
        //     mergeLocal,
        //     tableLocal,
        //     tempLocal0
        // );
        // CopyOut(blockId);
    }

private:
    __aicore__ inline void CopyIn_ms(uint32_t datablockId)
    {
        LocalTensor<T> ms_inLocal = ms_inQueue.AllocTensor<T>();
        DataCopy(ms_inLocal, ms_input[datablockId * (computeNum / 4)], computeNum / 4);
        ms_inQueue.EnQue(ms_inLocal);
    }
    __aicore__ inline void CopyIn_mbl(uint32_t datablockId)
    {
        LocalTensor<T> mbl_inLocal = mbl_inQueue.AllocTensor<T>();
        DataCopy(mbl_inLocal, mbl_input[datablockId * (tileNum / 8)], tileNum / 8);
        mbl_inQueue.EnQue(mbl_inLocal);
    }

    __aicore__ inline void Compute(int32_t i,
                                   uint64_t tempNum,
                                   uint32_t &accCompressed,
                                   LocalTensor<T> &e_inLocal,// 32kb
                                   LocalTensor<T> &cmblLocal,// 2kb
                                   LocalTensor<T> &mergeLocal,// 32kb
                                   LocalTensor<T> &mblcmpLocal,// 4kb
                                   LocalTensor<T> &takeLocal,// 64b
                                   LocalTensor<T> &tableLocal,// 1kb
                                   LocalTensor<T> &tempLocal0,// 32kb
                                   LocalTensor<T> &tempLocal2,// 2kb
                                   LocalTensor<T> &offset0Local,// 32b
                                   LocalTensor<T> &offset1Local,// 32b
                                   LocalTensor<T> &tempLocal3,// 除数，取数掩码, 2kb
                                   LocalTensor<T> &mask1Local,// 2kb
                                   LocalTensor<T> &mask15Local,// 2kb
                                   LocalTensor<float> &mask257Local,// 2kb
                                   LocalTensor<float> &maskne1Local,// 2kb
                                   LocalTensor<float> &src1Float, // 2kb
                                   LocalTensor<float> &dst1Float // 2kb
                                )
    {
        LocalTensor<T> ms_inLocal = ms_inQueue.DeQue<T>();
        LocalTensor<T> mbl_inLocal = mbl_inQueue.DeQue<T>();
        LocalTensor<T> outLocal = outQueue.AllocTensor<T>();

        AIV_WITH_BARRIER(ShiftRight, mbl_inLocal[tileNum / 8], mbl_inLocal, (uint32_t)16, tileNum / 8);
        AIV_WITH_BARRIER(ShiftRight, mbl_inLocal[tileNum / 4], mbl_inLocal, (uint32_t)8, tileNum / 4);
        AIV_WITH_BARRIER(ShiftRight, mbl_inLocal[tileNum / 2], mbl_inLocal, (uint32_t)4, tileNum / 2);
        AIV_WITH_BARRIER(And, mbl_inLocal, mbl_inLocal, mask15Local, tileNum * 2);
        // 400GB/s
        // DumpTensor(mbl_inLocal, 1, 256);

        // // 计算divLocal与fetchLocal
        // AIV_WITH_BARRIER(Cast, div_fetchLocal.template ReinterpretCast<float>(), mbl_inLocal.template ReinterpretCast<int32_t>(), RoundMode::CAST_TRUNC, tileNum);
        // AIV_WITH_BARRIER(Power, tempLocal2.template ReinterpretCast<float>(), (float)2, div_fetchLocal.template ReinterpretCast<float>(), tileNum);
        // AIV_WITH_BARRIER(FusedMulAdd, tempLocal2.template ReinterpretCast<float>(), mask257Local.template ReinterpretCast<float>(), maskne1Local.template ReinterpretCast<float>(), tileNum);
        // AIV_WITH_BARRIER(Cast, div_fetchLocal.template ReinterpretCast<int32_t>(), tempLocal2.template ReinterpretCast<float>(), RoundMode::CAST_TRUNC, tileNum);
        // 284Gb/s

        // 比较获得掩码并得到对应的整数掩码
        AIV_WITH_BARRIER(Compare, mblcmpLocal.template ReinterpretCast<uint8_t>(), cmblLocal.template ReinterpretCast<float>(), mbl_inLocal.template ReinterpretCast<float>(), CMPMODE::LT, tileNum);
        AIV_WITH_BARRIER(Select, tempLocal0.template ReinterpretCast<float>(), mblcmpLocal, mask1Local.template ReinterpretCast<float>(), (float)0, SELMODE::VSEL_TENSOR_SCALAR_MODE, tileNum);
        
        // 更新cmbl用于下一次计算
        // DumpTensor(cmblLocal, 1, 256);
        AIV_WITH_BARRIER(Adds, cmblLocal.template ReinterpretCast<int32_t>(), cmblLocal.template ReinterpretCast<int32_t>(), (int32_t)16, (int32_t)tileNum);
        AIV_WITH_BARRIER(Sub, cmblLocal.template ReinterpretCast<float>(), cmblLocal.template ReinterpretCast<float>(), mbl_inLocal.template ReinterpretCast<float>(), tileNum);
        AIV_WITH_BARRIER(And, cmblLocal, cmblLocal, mask15Local, tileNum * 2);
        // DumpTensor(cmblLocal, 1, 256);

        Adds(mbl_inLocal.template ReinterpretCast<int32_t>(), mbl_inLocal.template ReinterpretCast<int32_t>(), (int32_t)(127), tileNum);
        ShiftLeft(mbl_inLocal, mbl_inLocal, (uint32_t)23, tileNum);
        Cast(mbl_inLocal.template ReinterpretCast<int32_t>(), mbl_inLocal.template ReinterpretCast<float>(), RoundMode::CAST_TRUNC, tileNum);
        // AIV_WITH_BARRIER(ShiftLeft, mbl_inLocal, mbl_inLocal, (uint32_t)2, tileNum);
        // // DumpTensor(mbl_inLocal, 1, 256);
        // AIV_WITH_BARRIER(Gather, mbl_inLocal, takeLocal, mbl_inLocal, 0, tileNum);
        // DumpTensor(mbl_inLocal, 1, 256);
        // 336GB/s
        // DumpTensor(mbl_inLocal, 1, 256);

        // 331GB/s

        // 计算tileNum长度的前缀和
        // static constexpr CumSumConfig cumSumConfig{true, false, false};
        auto src0Float = tempLocal0.template ReinterpretCast<float>();
        auto dst0Float = outLocal.template ReinterpretCast<float>();
        auto lastRowFloat = mergeLocal.template ReinterpretCast<float>();
        auto sharedTmp = e_inLocal[8].template ReinterpretCast<uint8_t>();
        // const CumSumInfo cumSumInfo0{
        //     // 8, 
        //     // 64
        //     // 1, tileNum
        //     // 16, 
        //     // 32
        //     32,
        //     16
        // };
        AIV_WITH_BARRIER((CumSum<float, cumSumConfig>), dst0Float, lastRowFloat, src0Float, sharedTmp, cumSumInfo0);
        // DumpTensor(outLocal, 1, tileNum);
        // 305GB

        DataCopy(e_inLocal[8], outLocal, tileNum);
        AIV_WITH_BARRIER(Add, e_inLocal[8 + 8].template ReinterpretCast<int32_t>(), e_inLocal[8].template ReinterpretCast<int32_t>(), e_inLocal[8 + 8].template ReinterpretCast<int32_t>(), tileNum - 8);
        AIV_WITH_BARRIER(Add, e_inLocal[8 + 16].template ReinterpretCast<int32_t>(), e_inLocal[8].template ReinterpretCast<int32_t>(), e_inLocal[8 + 16].template ReinterpretCast<int32_t>(), tileNum - 16);
        AIV_WITH_BARRIER(Add, e_inLocal[8 + 32].template ReinterpretCast<int32_t>(), e_inLocal[8].template ReinterpretCast<int32_t>(), e_inLocal[8 + 32].template ReinterpretCast<int32_t>(), tileNum - 32);
        AIV_WITH_BARRIER(Add, e_inLocal[8 + 64].template ReinterpretCast<int32_t>(), e_inLocal[8].template ReinterpretCast<int32_t>(), e_inLocal[8 + 64].template ReinterpretCast<int32_t>(), tileNum - 64);
        AIV_WITH_BARRIER(Add, e_inLocal[8 + 128].template ReinterpretCast<int32_t>(), e_inLocal[8].template ReinterpretCast<int32_t>(), e_inLocal[8 + 128].template ReinterpretCast<int32_t>(), tileNum - 128);
        AIV_WITH_BARRIER(Add, e_inLocal[8 + 256].template ReinterpretCast<int32_t>(), e_inLocal[8].template ReinterpretCast<int32_t>(), e_inLocal[8 + 256].template ReinterpretCast<int32_t>(), tileNum - 256);
        // if(i == datablockNum - 1)
        // {
        //     // DumpTensor(tempLocal0, 1, tileNum);
        //     DumpTensor(e_inLocal[8], 1, tileNum);
        //     // uint32_t totalNum = 0;
        //     // for(int j = 1; j < 64; j ++){
        //     //     totalNum += e_inLocal[8 + j * 8];
        //     // }
        // }

        AIV_WITH_BARRIER(GatherMask, tempLocal2.template ReinterpretCast<float>(), e_inLocal[8].template ReinterpretCast<float>(), offset0Local.template ReinterpretCast<uint32_t>(), true, computeNum, {1, 1, 1, 0}, tempNum);
        // DumpTensor(tempLocal2, 1, tileNum / 8);
        // 280GB/s
        // if(i == datablockNum - 1)
        // {
        //     // DumpTensor(tempLocal0, 1, tileNum);
        //     // DumpTensor(e_inLocal[8], 1, tileNum);
        //     DumpTensor(tempLocal2, 1, tileNum / 8);
        //     // uint32_t totalNum = 0;
        //     // for(int j = 1; j < 64; j ++){
        //     //     totalNum += e_inLocal[8 + j * 8];
        //     // }
        // }

        // auto src1FLoat = tempLocal2.template ReinterpretCast<float>();
        // auto dst1Float = tempLocal3.template ReinterpretCast<float>();
        // const CumSumInfo cumSumInfo1{1, 32};
        // AIV_WITH_BARRIER((CumSum<float, cumSumConfig>), dst1Float, lastRowFloat, src1Float, sharedTmp, cumSumInfo1);
        // 230GB/s

        AIV_WITH_BARRIER((Broadcast<float, 2, 1>), tempLocal0.template ReinterpretCast<float>(), tempLocal2.template ReinterpretCast<float>(), dstShape_1, srcShape_1);
        // DumpTensor(tempLocal0, 1, 256);
        AIV_WITH_BARRIER(Add, outLocal[8].template ReinterpretCast<int32_t>(), outLocal[8].template ReinterpretCast<int32_t>(), tempLocal0.template ReinterpretCast<int32_t>(), tileNum - 8);
        // AIV_WITH_BARRIER(Add, outLocal[16].template ReinterpretCast<int32_t>(), outLocal[16].template ReinterpretCast<int32_t>(), tempLocal0.template ReinterpretCast<int32_t>(), tileNum - 16);
        // DumpTensor(outLocal, 1, 256);
        // 225GB/s
        // if(i == datablockNum - 1)
        // {
        //     // DumpTensor(tempLocal0, 1, tileNum);
        //     // DumpTensor(e_inLocal[8], 1, tileNum);
        //     // DumpTensor(tempLocal2, 1, tileNum / 8);
        //     DumpTensor(outLocal, 1, tileNum);
        //     // uint32_t totalNum = 0;
        //     // for(int j = 1; j < 64; j ++){
        //     //     totalNum += e_inLocal[8 + j * 8];
        //     // }
        // }

        // 将tileNum的前缀和扩展为computeNum长度
        AIV_WITH_BARRIER((Broadcast<float, 2, 0>), tempLocal0.template ReinterpretCast<float>(), outLocal.template ReinterpretCast<float>(), dstShape_prefix, srcShape_prefix);
        // DumpTensor(tempLocal0, 1, 1024);
        AIV_WITH_BARRIER(Muls, tempLocal2.template ReinterpretCast<int32_t>(), offset1Local.template ReinterpretCast<int32_t>(), (int32_t)outLocal(tileNum - 1), tileLength);
        // DumpTensor(tempLocal2, 1, 256);
        AIV_WITH_BARRIER((Broadcast<float, 2, 1>), outLocal.template ReinterpretCast<float>(), tempLocal2.template ReinterpretCast<float>(), dstShape_offset, srcShape_offset);
        // DumpTensor(outLocal, 1, 1024);
        AIV_WITH_BARRIER(Add, outLocal.template ReinterpretCast<int32_t>(), tempLocal0.template ReinterpretCast<int32_t>(), outLocal.template ReinterpretCast<int32_t>(), computeNum);
        // 得到computeNum长度的掩码
        AIV_WITH_BARRIER((Broadcast<float, 2, 0>), mblcmpLocal.template ReinterpretCast<float>(), mblcmpLocal.template ReinterpretCast<float>(), dstShape_mblcmp, srcShape_mblcmp);
        // 186GB/s
        // DumpTensor(outLocal, 1, 256);

        // 读取需要的码字
        SCALAR_WITH_BARRIER(totalCompressed = outLocal(computeNum - 1) / 2);
        AIV_WITH_BARRIER(DataCopy, e_inLocal[8], e_input[accCompressed], totalCompressed);
        int32_t eventIDMTE2ToV = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
        SCALAR_WITH_BARRIER(accCompressed = accCompressed + totalCompressed);
        // 164GB/s
        
        // 计算反向gather的索引
        // AIV_WITH_BARRIER(ShiftLeft, outLocal, outLocal, (uint32_t)2, computeNum);
        // AIV_WITH_BARRIER(Adds, outLocal.template ReinterpretCast<int32_t>(), outLocal.template ReinterpretCast<int32_t>(), (int32_t)28, computeNum);
        // AIV_WITH_BARRIER(Select, outLocal.template ReinterpretCast<float>(), mblcmpLocal, outLocal.template ReinterpretCast<float>(),
        //        (float)0, SELMODE::VSEL_TENSOR_TENSOR_MODE, computeNum);
        // 153GB/s
        // DumpTensor(outLocal, 1, 256);

        AIV_WITH_BARRIER(ShiftLeft, outLocal, outLocal, (uint32_t)1, computeNum);
        AIV_WITH_BARRIER(Adds, outLocal.template ReinterpretCast<int32_t>(), outLocal.template ReinterpretCast<int32_t>(), (int32_t)30, computeNum);

        // AIV_WITH_BARRIER(ShiftRight, e_inLocal[8 + totalCompressed], e_inLocal[8], (uint32_t)16, totalCompressed);
        // AIV_WITH_BARRIER(ShiftLeft, e_inLocal[8], e_inLocal[8], (uint32_t)16, totalCompressed);
        // AIV_WITH_BARRIER(ShiftRight, e_inLocal[8], e_inLocal[8], (uint32_t)16, totalCompressed);
        // 151GB/s
        // DumpTensor(e_inLocal, 1, 256);

        AIV_WITH_BARRIER(Gather, outLocal.template ReinterpretCast<float>(), e_inLocal.template ReinterpretCast<float>(), outLocal, (uint32_t)0, (uint32_t)computeNum);
        AIV_WITH_BARRIER(ShiftRight, outLocal.template ReinterpretCast<uint16_t>(), outLocal.template ReinterpretCast<uint16_t>(), (uint16_t)16, computeNum);
        // DumpTensor(outLocal, 1, 1024);
        // 恢复缓冲区
        AIV_WITH_BARRIER(ShiftLeft, tempLocal0, mergeLocal, (uint32_t)16, computeNum);
        AIV_WITH_BARRIER(Or, tempLocal0.template ReinterpretCast<uint16_t>(), tempLocal0.template ReinterpretCast<uint16_t>(), outLocal.template ReinterpretCast<uint16_t>(), computeNum * 2);
        AIV_WITH_BARRIER(Select, mergeLocal.template ReinterpretCast<float>(), mblcmpLocal, tempLocal0.template ReinterpretCast<float>(),
               mergeLocal.template ReinterpretCast<float>(), SELMODE::VSEL_TENSOR_TENSOR_MODE, computeNum);

        // 提取对应的码字
        // AIV_WITH_BARRIER(Adds, outLocal.template ReinterpretCast<int32_t>(), mbl_inLocal.template ReinterpretCast<int32_t>(), (int32_t)(-1), tileNum);
        AIV_WITH_BARRIER((Broadcast<float, 2, 0>), tempLocal0.template ReinterpretCast<float>(), mbl_inLocal.template ReinterpretCast<float>(), dstShape_div_fetch, srcShape_div_fetch);
        // AIV_WITH_BARRIER(ShiftRight, outLocal, tempLocal0, (uint32_t)14, computeNum);
        AIV_WITH_BARRIER(Adds, outLocal.template ReinterpretCast<int32_t>(), tempLocal0.template ReinterpretCast<int32_t>(), (int32_t)(-1), computeNum);
        // DumpTensor(outLocal, 1, computeNum / 8);
        AIV_WITH_BARRIER(And, outLocal.template ReinterpretCast<uint16_t>(), mergeLocal.template ReinterpretCast<uint16_t>(), outLocal.template ReinterpretCast<uint16_t>(), computeNum * 2);
        // DumpTensor(outLocal, 1, computeNum / 8);

        // 反向table替换
        AIV_WITH_BARRIER(ShiftLeft, outLocal, outLocal, (uint32_t)2, computeNum);
        AIV_WITH_BARRIER(Gather, outLocal, tableLocal, outLocal, (uint32_t)0, (uint32_t)computeNum);

        // 更新mergeLocal，建议把除法变成减法
        // AIV_WITH_BARRIER(ShiftLeft, tempLocal0, tempLocal0, (uint32_t)18, computeNum);
        // AIV_WITH_BARRIER(ShiftRight, tempLocal0, tempLocal0, (uint32_t)23, computeNum);
        AIV_WITH_BARRIER(Div, mergeLocal.template ReinterpretCast<float>(), mergeLocal.template ReinterpretCast<float>(), tempLocal0.template ReinterpretCast<float>(), (int32_t)computeNum);
        AIV_WITH_BARRIER(Cast, mergeLocal.template ReinterpretCast<int32_t>(), mergeLocal.template ReinterpretCast<float>(), RoundMode::CAST_TRUNC, computeNum);

        // 恢复原始数据，尾数，符号，指数进行组合
        AIV_WITH_BARRIER(ShiftRight, (tempLocal0.template ReinterpretCast<uint16_t>())[computeNum / 2], ms_inLocal.template ReinterpretCast<uint16_t>(), (uint16_t)8, computeNum / 2);
        AIV_WITH_BARRIER(ShiftLeft, tempLocal0.template ReinterpretCast<uint16_t>(), ms_inLocal.template ReinterpretCast<uint16_t>(), (uint16_t)8, computeNum / 2);
        AIV_WITH_BARRIER(ShiftRight, tempLocal0.template ReinterpretCast<uint16_t>(), tempLocal0.template ReinterpretCast<uint16_t>(), (uint16_t)8, computeNum / 2);

        AIV_WITH_BARRIER(ShiftLeft, outLocal, outLocal, (uint32_t)8, computeNum / 2);
        AIV_WITH_BARRIER(ShiftLeft, outLocal[computeNum / 2], outLocal[computeNum / 2], (uint32_t)24, computeNum / 2);
        AIV_WITH_BARRIER(Or, outLocal.template ReinterpretCast<uint16_t>(), outLocal.template ReinterpretCast<uint16_t>(), outLocal[computeNum / 2].template ReinterpretCast<uint16_t>(), computeNum / 2 * 2);

        AIV_WITH_BARRIER(Or, outLocal.template ReinterpretCast<uint16_t>(), outLocal.template ReinterpretCast<uint16_t>(), tempLocal0.template ReinterpretCast<uint16_t>(), computeNum * 2);

        // AIV_WITH_BARRIER(ShiftLeft, tempLocal0.template ReinterpretCast<uint16_t>(), outLocal.template ReinterpretCast<uint16_t>(), (uint16_t)15, computeNum);
        // AIV_WITH_BARRIER(ShiftRight, outLocal.template ReinterpretCast<uint16_t>(), outLocal.template ReinterpretCast<uint16_t>(), (uint16_t)1, computeNum);
        // AIV_WITH_BARRIER(Or, outLocal.template ReinterpretCast<uint16_t>(), outLocal.template ReinterpretCast<uint16_t>(), tempLocal0.template ReinterpretCast<uint16_t>(), computeNum);

        outQueue.EnQue(outLocal);
        ms_inQueue.FreeTensor(ms_inLocal);
        mbl_inQueue.FreeTensor(mbl_inLocal);
    }

    __aicore__ inline void ComputeFirst(
        uint32_t &accCompressed,
        LocalTensor<T> &mergeLocal,
        LocalTensor<T> &tableLocal,
        LocalTensor<T> &tempLocal0
    )
    {
        LocalTensor<T> ms_inLocal = ms_inQueue.DeQue<T>();
        LocalTensor<T> outLocal = outQueue.AllocTensor<T>();

        AIV_WITH_BARRIER(ShiftLeft, outLocal, mergeLocal, (uint32_t)2, computeNum);
        AIV_WITH_BARRIER(Gather, outLocal, tableLocal, outLocal, (uint32_t)0, (uint32_t)computeNum);

        AIV_WITH_BARRIER(ShiftRight, (tempLocal0.template ReinterpretCast<uint16_t>())[computeNum / 2], ms_inLocal.template ReinterpretCast<uint16_t>(), (uint16_t)8, computeNum / 2);
        AIV_WITH_BARRIER(ShiftLeft, tempLocal0.template ReinterpretCast<uint16_t>(), ms_inLocal.template ReinterpretCast<uint16_t>(), (uint16_t)8, computeNum / 2);
        AIV_WITH_BARRIER(ShiftRight, tempLocal0.template ReinterpretCast<uint16_t>(), tempLocal0.template ReinterpretCast<uint16_t>(), (uint16_t)8, computeNum / 2);

        AIV_WITH_BARRIER(ShiftLeft, outLocal, outLocal, (uint32_t)8, computeNum / 2);
        AIV_WITH_BARRIER(ShiftLeft, outLocal[computeNum / 2], outLocal[computeNum / 2], (uint32_t)24, computeNum / 2);
        AIV_WITH_BARRIER(Or, outLocal.template ReinterpretCast<uint16_t>(), outLocal.template ReinterpretCast<uint16_t>(), outLocal[computeNum / 2].template ReinterpretCast<uint16_t>(), computeNum / 2 * 2);

        AIV_WITH_BARRIER(Or, outLocal.template ReinterpretCast<uint16_t>(), outLocal.template ReinterpretCast<uint16_t>(), tempLocal0.template ReinterpretCast<uint16_t>(), computeNum * 2);

        AIV_WITH_BARRIER(ShiftLeft, tempLocal0.template ReinterpretCast<uint16_t>(), outLocal.template ReinterpretCast<uint16_t>(), (uint16_t)15, computeNum);
        AIV_WITH_BARRIER(ShiftRight, outLocal.template ReinterpretCast<uint16_t>(), outLocal.template ReinterpretCast<uint16_t>(), (uint16_t)1, computeNum);
        AIV_WITH_BARRIER(Or, outLocal.template ReinterpretCast<uint16_t>(), outLocal.template ReinterpretCast<uint16_t>(), tempLocal0.template ReinterpretCast<uint16_t>(), computeNum);

        outQueue.EnQue(outLocal);
        ms_inQueue.FreeTensor(ms_inLocal);
    }

    __aicore__ inline void CopyOut(uint32_t datablockId)
    {
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
    TBuf<TPosition::VECCALC> offset0;
    TBuf<TPosition::VECCALC> offset1;
    TBuf<TPosition::VECCALC> div_fetch;
    TBuf<TPosition::VECCALC> mask1;
    TBuf<TPosition::VECCALC> mask15;
    TBuf<TPosition::VECCALC> mask257;
    TBuf<TPosition::VECCALC> maskne1;

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

    // uint32_t srcShape_0[2];
    // uint32_t dstShape_0[2];
    uint32_t srcShape_1[2];
    uint32_t dstShape_1[2];
    uint32_t dstShape_prefix[2];
    uint32_t srcShape_prefix[2];
    uint32_t dstShape_offset[2];
    uint32_t srcShape_offset[2];
    uint32_t dstShape_mblcmp[2];
    uint32_t srcShape_mblcmp[2];
    uint32_t dstShape_div_fetch[2];
    uint32_t srcShape_div_fetch[2];

    static constexpr CumSumConfig cumSumConfig{true, false, false};
    const CumSumInfo cumSumInfo0{
        // 8, 
        // 64
        // 1, tileNum
        // 16, 
        // 32
        64,
        8
    };
    // const CumSumInfo cumSumInfo0{
    //     // 8, 
    //     // 64
    //     // 1, tileNum
    //     // 16, 
    //     // 32
    //     32,
    //     16
    // };
    const CumSumInfo cumSumInfo1{1, 32};

};

__global__ __aicore__ void decompBF16(
    uint32_t BUFFER_NUM,
    uint32_t elementNum,
    uint32_t tileLength,
    uint32_t tileNum,
    uint32_t threadblockNum,
    uint32_t datablockNum,
    uint32_t datablockSize,
    __gm__ uint8_t* eGlobal,
    __gm__ uint8_t* tableGlobal,
    __gm__ uint8_t* msGlobal,
    __gm__ uint8_t* mblGlobal,
    __gm__ uint8_t* compSizePrefix,
    __gm__ uint8_t* decompressedGlobal)
{
    TPipe pipe;
    DecompressKernelBF16<uint32_t> op;
    op.Init(&pipe, BUFFER_NUM, elementNum, tileLength, tileNum, threadblockNum, datablockNum, datablockSize,
            eGlobal, tableGlobal, msGlobal, mblGlobal, compSizePrefix, decompressedGlobal);
    op.Process();
}

extern "C" void enec_decompress(Header* cphd, void* stream, uint8_t* compressed, uint8_t* decompressed)
{
    switch (cphd->dataType)
    {
    case 0:
    { // BF16
        uint32_t elementNum = cphd->dataBlockSize / sizeof(uint16_t);
        uint32_t tileNum = elementNum / cphd->tileLength;
        decompBF16<<<cphd->threadBlockNum, nullptr, stream>>>(1, elementNum, cphd->tileLength, tileNum, cphd->threadBlockNum, cphd->dataBlockNum, cphd->dataBlockSize,
                                                            getCompressed_exp(cphd, compressed), getTable(cphd, compressed), getMsdata(cphd, compressed), getMbl(cphd, compressed), getCompSizePrefix(cphd, compressed), decompressed);
        break;
    }
    case 1:
    { // FP16

        break;
    }
    case 2:
    { // FP32

        break;
    }
    default:
    {

        return;
    }
    }
}

