#include "hans_utils.h"
#include "kernel_operator.h"

using namespace AscendC;

template<typename T>// uint32_t
class DecompressKernelBF16 {
public:
    __aicore__ inline DecompressKernelBF16() {}
    // 输入：指数数组（uint8_t），32位编码表，
    // 输出：max_bit_length数组（4bits * tile_num * 2），码字（max_bit_length * tile_len,最终的压缩块大小需要向上取到32的倍数，满足32字节对齐的要求）

    __aicore__ inline void Init(TPipe* pipe,
                                uint32_t BUFFER_NUM,
                                uint32_t elementNum,
                                uint32_t tileLength,
                                uint32_t tileNum,
                                uint32_t threadblockNum,
                                uint32_t datablockNum,
                                uint32_t datablockSize,
                                __gm__ uint8_t* eGlobal, //e_input
                                __gm__ uint8_t* tableGlobal, //table_input
                                __gm__ uint8_t* msGlobal, // ms_input
                                __gm__ uint8_t* mblGlobal, // mbl_input
                                __gm__ uint8_t* compSizePrefix, // compSizePrefix
                                __gm__ uint8_t* decompressedGlobal //output
                                ) {
        this->pipe = pipe;
        this->blockId = GetBlockIdx(); //获取当前blockId
        this->blockNum = GetBlockNum(); //获取当前blockNum
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

        table_input.SetGlobalBuffer((__gm__ T*)(tableGlobal));
        ms_input.SetGlobalBuffer((__gm__ T*)(msGlobal));
        mbl_input.SetGlobalBuffer((__gm__ T*)(mblGlobal));
        compSizePrefix_input.SetGlobalBuffer((__gm__ T*)(compSizePrefix));
        output.SetGlobalBuffer((__gm__ T*)(decompressedGlobal));
        // e_input.SetGlobalBuffer((__gm__ T*)(eGlobal));

        pipe->InitBuffer(outQueue, BUFFER_NUM, computeNum * sizeof(T));// 初始化Pipe缓冲区，每个Tile大小TILE_LEN, 32kb
        pipe->InitBuffer(ms_inQueue, BUFFER_NUM, computeNum);// 字节为单位，8kb
        pipe->InitBuffer(mbl_inQueue, BUFFER_NUM, tileNum * sizeof(T));// 初始化Pipe缓冲区，每个Tile大小TILE_LEN, 2kb

        pipe->InitBuffer(compPrefix, BLOCK_NUM * sizeof(T));// 256 * 4bytes 1kb
        LocalTensor<T> compPrefixLocal = compPrefix.Get<T>();
        DataCopy(compPrefixLocal, compSizePrefix_input, BLOCK_NUM);
        e_input.SetGlobalBuffer((__gm__ T*)(eGlobal + compPrefixLocal(blockId)));
    }

    __aicore__ inline void Process(//uint32_t& compressedSize
                                    ) {

        pipe->InitBuffer(e_in, computeNum * sizeof(T) + 32);// 32kb
        pipe->InitBuffer(cmbl, tileNum * sizeof(T));// 2kb
        pipe->InitBuffer(merge, computeNum * sizeof(T));// 32kb
        pipe->InitBuffer(mblcmp, tileNum * sizeof(T));// 2kb
        pipe->InitBuffer(take, tileNum * sizeof(T));// 2kb
        pipe->InitBuffer(table, HISTOGRAM_BINS * sizeof(T));// 1kb
        pipe->InitBuffer(table8, HISTOGRAM_BINS);// 256b
        pipe->InitBuffer(temp0, computeNum * sizeof(T));// 32kb   
        pipe->InitBuffer(temp1, computeNum * sizeof(T));// 32kb
        pipe->InitBuffer(temp2, tileNum * sizeof(T));// 2kb
        pipe->InitBuffer(temp3, tileNum * sizeof(T));// 2kb
        pipe->InitBuffer(offset, 256 * sizeof(T));// 1kb
        pipe->InitBuffer(mask15, tileNum * sizeof(T));// 2kb
        
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
        Duplicate(mask15Local, (T)15, tileNum);// 00000000 00000000 00000000 00001111

        for(int i = 0; i < 256; i ++){
            if(i % 2 == 0)
                offsetLocal(i) = (uint32_t)0;
            else 
                offsetLocal(i) = (uint32_t)2147483648;
        }

        // 需要改成实际的e_input地址
        DataCopy(mergeLocal, e_input, computeNum / 2);
        int32_t eventIDMTE2ToV0 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV0);
        WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV0);
        ShiftRight(mergeLocal[computeNum / 2], mergeLocal, (uint32_t)16, computeNum / 2);
        PipeBarrier<PIPE_ALL>();
        ShiftLeft(tempLocal0, mergeLocal, (uint32_t)16, computeNum / 2);
        PipeBarrier<PIPE_ALL>();
        ShiftRight(mergeLocal, tempLocal0, (uint32_t)16, computeNum / 2);
        PipeBarrier<PIPE_ALL>();
        // mergeLocal初始化和压缩的一致

        DataCopy(cmblLocal, mbl_input[blockId * tileNum / 8], tileNum / 8);
        int32_t eventIDMTE2ToV1 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV1);
        WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV1);
        ShiftRight(cmblLocal[tileNum / 8], cmblLocal, (uint32_t)16, tileNum / 8);
        PipeBarrier<PIPE_ALL>();
        ShiftRight(cmblLocal[tileNum / 4], cmblLocal, (uint32_t)8, tileNum / 4);
        PipeBarrier<PIPE_ALL>();
        ShiftRight(cmblLocal[tileNum / 2], cmblLocal, (uint32_t)4, tileNum / 2);
        PipeBarrier<PIPE_ALL>();
        And(cmblLocal, cmblLocal, mask15Local, tileNum * 2);
        PipeBarrier<PIPE_ALL>();
        // cmblLocal初始化和压缩结果一致

        for(int i = 0; i < 8; i ++){
            uint32_t extra = (1 << i) - 1;
            uint32_t divNum = 1 << i;
            uint32_t mbl = i;
            takeLocal(i) = (extra << 14) | (divNum << 5) | (mbl);
        }

        DataCopy(table8Local.template ReinterpretCast<T>(), table_input, HISTOGRAM_BINS / sizeof(T));
        int32_t eventIDMTE2ToV2 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV2);
        WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV2);
        for(int i = 0; i < HISTOGRAM_BINS; i ++){
            tableLocal(i) = (uint32_t)table8Local(i);
        }
        
        // blockNum = 256;
        uint32_t accCompressed =  computeNum * sizeof(uint16_t) / sizeof(T);// 32bits为单位
        int32_t remainderNum = datablockNum % blockNum;
        // assert(remainderNum == 1);
        int32_t remainderStart = datablockNum - remainderNum;
        // assert(remainderStart == 1024);
        int32_t startdataBlock = blockId < remainderNum ? remainderStart + blockId : (remainderStart - blockNum) + blockId;
        // assert(startdataBlock == 1024);
        // assert(blockId == 0);
        // assert(blockNum == 256);
        // int num = 0;
        // if(blockId == 1)
        // assert(startdataBlock == 1024);
        for(int32_t i = startdataBlock; i > (int32_t)blockId 
        // + 
        // (int32_t)blockNum * 0
        ; i -= (int32_t)blockNum){
        // num = num + 1;
        // assert(i == 1024);
        // assert(i >= 0 && i < datablockNum);
        // if(blockId == 0){
        // int i = 1024;
            CopyIn_mbl(i);
            CopyIn_ms(i);
            Compute(
                    i,
                    accCompressed,
                    e_inLocal,
                    cmblLocal, 
                    mergeLocal,
                    mblcmpLocal, 
                    takeLocal,
                    tableLocal,
                    tempLocal0,
                    tempLocal1,
                    tempLocal2,
                    tempLocal3,
                    offsetLocal,
                    mask15Local
                    );
            CopyOut(i);// 拷贝结果到GM，这里拷贝到字节数必定为32字节的倍数
        }
        // assert(num == 1);
        // }
        CopyIn_ms(blockId);
        ComputeFirst(
                accCompressed,
                e_inLocal,
                cmblLocal, 
                mergeLocal,
                mblcmpLocal, 
                takeLocal,
                tableLocal,
                tempLocal0,
                tempLocal1,
                tempLocal2,
                tempLocal3,
                offsetLocal,
                mask15Local
        );// 解压第一个datablock的数据
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

    __aicore__ inline void Compute( int32_t i,
                                    uint32_t& accCompressed,       
                                    LocalTensor<T>& e_inLocal,// 32 + 32kb
                                    LocalTensor<T>& cmblLocal,// 2kb
                                    LocalTensor<T>& mergeLocal,// 32kb
                                    LocalTensor<T>& mblcmpLocal,// 2kb
                                    LocalTensor<T>& takeLocal,// 2kb
                                    LocalTensor<T>& tableLocal,// 2kb
                                    LocalTensor<T>& tempLocal0,// 32kb
                                    LocalTensor<T>& tempLocal1,// 32kb
                                    LocalTensor<T>& tempLocal2,// 2kb
                                    LocalTensor<T>& tempLocal3,// 2kb
                                    LocalTensor<T>& offsetLocal,// 2kb
                                    LocalTensor<T>& mask15Local// 2kb
    ) {
        LocalTensor<T> ms_inLocal = ms_inQueue.DeQue<T>();// 16kb
        LocalTensor<T> mbl_inLocal = mbl_inQueue.DeQue<T>();// 2kb
        LocalTensor<T> outLocal = outQueue.AllocTensor<T>();// 32kb
        //必须注意：当and和or的操作类型为uint32_t，需要将操作数量翻倍
        // 复原mbl
        // if(i == 512){
        //     DumpTensor(mbl_inLocal, 1, 512 / 8);
        // }
        // PipeBarrier<PIPE_ALL>();
        // ShiftRight(mbl_inLocal[tileNum / 8], mbl_inLocal, (uint32_t)16, tileNum / 8);
        // PipeBarrier<PIPE_ALL>();
        PipeBarrier<PIPE_ALL>();
        ShiftRight(mbl_inLocal[tileNum / 8], mbl_inLocal, (uint32_t)16, tileNum / 8);
        PipeBarrier<PIPE_ALL>();
        ShiftLeft(mbl_inLocal, mbl_inLocal, (uint32_t)16, tileNum / 4);
        PipeBarrier<PIPE_ALL>(); 
        ShiftRight(mbl_inLocal, mbl_inLocal, (uint32_t)16, tileNum / 4);
        PipeBarrier<PIPE_ALL>();
        // if(i == 768){
        //     DumpTensor(mbl_inLocal, 1, 128);
        // }
        // ShiftRight(mbl_inLocal[tileNum / 4], mbl_inLocal, (uint32_t)8, tileNum / 4);
        // PipeBarrier<PIPE_ALL>();
        ShiftRight(mbl_inLocal[tileNum / 4], mbl_inLocal, (uint32_t)8, tileNum / 4);
        PipeBarrier<PIPE_ALL>();
        ShiftLeft(mbl_inLocal, mbl_inLocal, (uint32_t)24, tileNum / 2);
        PipeBarrier<PIPE_ALL>();
        ShiftRight(mbl_inLocal, mbl_inLocal, (uint32_t)24, tileNum / 2);
        PipeBarrier<PIPE_ALL>();
        // if(i == 768){
        //     DumpTensor(mbl_inLocal, 1, 256);
        // }
        // ShiftRight(mbl_inLocal[tileNum / 2], mbl_inLocal, (uint32_t)4, tileNum / 2);
        // PipeBarrier<PIPE_ALL>();
        ShiftRight(mbl_inLocal[tileNum / 2], mbl_inLocal, (uint32_t)4, tileNum / 2);
        PipeBarrier<PIPE_ALL>();
        // And(mbl_inLocal, mbl_inLocal, mask15Local, tileNum * 2);
        // PipeBarrier<PIPE_ALL>();
        DataSyncBarrier<MemDsbT::ALL>();
        ShiftLeft(tempLocal0, mbl_inLocal, (uint32_t)28, tileNum / 2);
        PipeBarrier<PIPE_ALL>();
        DataSyncBarrier<MemDsbT::ALL>();
        ShiftRight(mbl_inLocal, tempLocal0, (uint32_t)28, tileNum / 2);
        PipeBarrier<PIPE_ALL>();
        DataSyncBarrier<MemDsbT::ALL>();
        // if(i == 769){
        //     DumpTensor(mbl_inLocal, 1, 512);
        // }
        // And(mbl_inLocal, mbl_inLocal, mask15Local, tileNum * 2);
        // PipeBarrier<PIPE_ALL>();

        // PipeBarrier<PIPE_ALL>();
        // ShiftRight(mbl_inLocal[tileNum / 8], mbl_inLocal, (uint32_t)16, tileNum / 8);
        // PipeBarrier<PIPE_ALL>();
        // ShiftLeft(mbl_inLocal, mbl_inLocal, (uint32_t)16, tileNum / 4);
        // PipeBarrier<PIPE_ALL>(); 
        // ShiftRight(mbl_inLocal, mbl_inLocal, (uint32_t)16, tileNum / 4);
        // PipeBarrier<PIPE_ALL>();
        // ShiftRight(mbl_inLocal[tileNum / 4], mbl_inLocal, (uint32_t)8, tileNum / 4);
        // PipeBarrier<PIPE_ALL>();
        // ShiftLeft(mbl_inLocal, mbl_inLocal, (uint32_t)24, tileNum / 2);
        // PipeBarrier<PIPE_ALL>();
        // ShiftRight(mbl_inLocal, mbl_inLocal, (uint32_t)24, tileNum / 2);
        // PipeBarrier<PIPE_ALL>();
        // ShiftRight(mbl_inLocal[tileNum / 2], mbl_inLocal, (uint32_t)4, tileNum / 2);
        // PipeBarrier<PIPE_ALL>();
        // ShiftLeft(mbl_inLocal, mbl_inLocal, (uint32_t)28, tileNum);
        // PipeBarrier<PIPE_ALL>();
        // ShiftLeft(mbl_inLocal, mbl_inLocal, (uint32_t)28, tileNum);
        // PipeBarrier<PIPE_ALL>();

        // for(int i = 0; i < 512; i ++){
        //     assert(mbl_inLocal(i) < 15);
        // }
        // 广播
        Broadcast<float, 2, 1>(tempLocal0.template ReinterpretCast<float>(), cmblLocal.template ReinterpretCast<float>(), dstShape_0, srcShape_0);
        PipeBarrier<PIPE_ALL>();

        // 更新cmbl，用于下一轮
        Adds(cmblLocal.template ReinterpretCast<int32_t>(), cmblLocal.template ReinterpretCast<int32_t>(), (int32_t)16, (int32_t)tileNum);
        PipeBarrier<PIPE_ALL>();
        Sub(cmblLocal.template ReinterpretCast<float>(), cmblLocal.template ReinterpretCast<float>(), mbl_inLocal.template ReinterpretCast<float>(), tileNum);
        PipeBarrier<PIPE_ALL>();
        // And(cmblLocal.template ReinterpretCast<uint16_t>(), cmblLocal.template ReinterpretCast<uint16_t>(), mask15Local.template ReinterpretCast<uint16_t>(), tileNum * 2); 
        ShiftLeft(cmblLocal, cmblLocal, (uint32_t)28, tileNum);
        ShiftRight(cmblLocal, cmblLocal, (uint32_t)28, tileNum);
        PipeBarrier<PIPE_ALL>();
        // if( i == 512){
        //     DumpTensor(cmblLocal, 1, 512);
        // }

        // 从mbl_inLocal获取提取码字掩码和右移的除数
        ShiftLeft(mbl_inLocal, mbl_inLocal, (uint32_t)2, tileNum);
        PipeBarrier<PIPE_ALL>();
        Gather(mbl_inLocal, takeLocal, mbl_inLocal, 0, tileNum);  
        PipeBarrier<PIPE_ALL>();

        Broadcast<float, 2, 1>(tempLocal1.template ReinterpretCast<float>(), mbl_inLocal.template ReinterpretCast<float>(), dstShape_0, srcShape_0);
        PipeBarrier<PIPE_ALL>();
        // int32_t eventIDVToV3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_V));
        // SetFlag<AscendC::HardEvent::V_V>(eventIDVToV3);
        // WaitFlag<AscendC::HardEvent::V_V>(eventIDVToV3);
        // DataCopy(outLocal, cmblLocal, computeNum);

        // 比较
        ShiftLeft(outLocal, tempLocal1, (uint32_t)27, computeNum);
        PipeBarrier<PIPE_ALL>();
        ShiftRight(outLocal, outLocal, (uint32_t)27, computeNum);
        PipeBarrier<PIPE_ALL>();
        Compare(mblcmpLocal.template ReinterpretCast<uint8_t>(), tempLocal0.template ReinterpretCast<float>(), outLocal.template ReinterpretCast<float>(), CMPMODE::LT, computeNum);
        PipeBarrier<PIPE_ALL>();
        Duplicate(tempLocal0, (T)1, computeNum);
        PipeBarrier<PIPE_ALL>();
        Select(tempLocal0.template ReinterpretCast<float>(), mblcmpLocal, tempLocal0.template ReinterpretCast<float>(), (float)0, SELMODE::VSEL_TENSOR_SCALAR_MODE, computeNum);  
        PipeBarrier<PIPE_ALL>();
        // if(i == 512){
        //     DumpTensor(mblcmpLocal, 1, 256);
        // }
        // 计算整数掩码的前缀和，得到待读取数目，进行datacopy
        static constexpr CumSumConfig cumSumConfig{true, false, false};
        auto src0FLoat = tempLocal0.template ReinterpretCast<float>();
        auto dst0Float = outLocal.template ReinterpretCast<float>();
        auto lastRowFloat = mergeLocal.template ReinterpretCast<float>();
        auto sharedTmp = e_inLocal[8].template ReinterpretCast<uint8_t>();
        PipeBarrier<PIPE_ALL>();
        const CumSumInfo cumSumInfo0{128, 64};
        PipeBarrier<PIPE_ALL>();
        CumSum<float, cumSumConfig>(
            dst0Float, 
            lastRowFloat, src0FLoat, sharedTmp, cumSumInfo0);
        PipeBarrier<PIPE_ALL>();

        uint64_t tempNum = 0;
        GatherMask(tempLocal2.template ReinterpretCast<float>(), outLocal.template ReinterpretCast<float>(), offsetLocal.template ReinterpretCast<uint32_t>(), true, computeNum, {1, 1, 1, 0}, tempNum);
        PipeBarrier<PIPE_ALL>();

        auto src1FLoat = tempLocal2.template ReinterpretCast<float>();
        auto dst1Float = tempLocal3.template ReinterpretCast<float>();      
        const CumSumInfo cumSumInfo1{1, 128};
        PipeBarrier<PIPE_ALL>();
        CumSum<float, cumSumConfig>(dst1Float, lastRowFloat, src1FLoat, sharedTmp, cumSumInfo1);
        PipeBarrier<PIPE_ALL>();
        Broadcast<float, 2, 1>(tempLocal1.template ReinterpretCast<float>(), tempLocal3.template ReinterpretCast<float>(), dstShape_1, srcShape_1);
        PipeBarrier<PIPE_ALL>();
        Add(outLocal[64].template ReinterpretCast<int32_t>(), outLocal[64].template ReinterpretCast<int32_t>(), tempLocal1.template ReinterpretCast<int32_t>(), computeNum - 64);
        PipeBarrier<PIPE_ALL>();
        // if(i == 768){
        //     DumpTensor(outLocal, 1, computeNum);
        // }
        uint32_t totalCompressed = outLocal(computeNum - 1) / 2;// 以32bits为单位
        PipeBarrier<PIPE_ALL>();
        DataCopy(e_inLocal[8], e_input[accCompressed], totalCompressed);
        int32_t eventIDMTE2ToV = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
        accCompressed = accCompressed + totalCompressed;// 以32bits为单位
        PipeBarrier<PIPE_ALL>();

        // 计算反向gather的整数类型掩码，注意是以字节为单位
        Adds(outLocal.template ReinterpretCast<int32_t>(), outLocal.template ReinterpretCast<int32_t>(), (int32_t)(-1), computeNum);
        PipeBarrier<PIPE_ALL>();
        ShiftLeft(outLocal, outLocal, (uint32_t)2, computeNum);
        PipeBarrier<PIPE_ALL>();
        Adds(outLocal.template ReinterpretCast<int32_t>(), outLocal.template ReinterpretCast<int32_t>(), (int32_t)32, computeNum);
        PipeBarrier<PIPE_ALL>();
        Mul(outLocal.template ReinterpretCast<int32_t>(), tempLocal0.template ReinterpretCast<int32_t>(), outLocal.template ReinterpretCast<int32_t>(), computeNum);
        PipeBarrier<PIPE_ALL>();

        // if(i == 512){
        //     DumpTensor(outLocal, 1, totalCompressed + 8);
        // }

        // if(i == 512){
        //     e_inLocal(0) = totalCompressed;
        //     DumpTensor(e_inLocal, 1, totalCompressed + 8);
        // }

        // 恢复e_inLocal数据
        ShiftRight(e_inLocal[8 + totalCompressed], e_inLocal[8], (uint32_t)16, totalCompressed);
        PipeBarrier<PIPE_ALL>();
        ShiftLeft(e_inLocal[8], e_inLocal[8], (uint32_t)16, totalCompressed);
        PipeBarrier<PIPE_ALL>();
        ShiftRight(e_inLocal[8], e_inLocal[8], (uint32_t)16, totalCompressed);
        PipeBarrier<PIPE_ALL>();

        Duplicate(e_inLocal, (T)0, 8);// 00000000 00000000 00000000 00001111
        PipeBarrier<PIPE_ALL>();
        // 使用反向gather恢复数据
        Gather(outLocal.template ReinterpretCast<float>(), e_inLocal.template ReinterpretCast<float>(), outLocal, (uint32_t)0, (uint32_t)computeNum);
        PipeBarrier<PIPE_ALL>();

        // if( i == 514){
        //     DumpTensor(mergeLocal, 1, 512);
        // }

        // 更新mergeLocal，通过上述的位掩码进行更新
        ShiftLeft(tempLocal0, mergeLocal, (uint32_t)16, computeNum);
        PipeBarrier<PIPE_ALL>();
        Or(tempLocal0.template ReinterpretCast<uint16_t>(), tempLocal0.template ReinterpretCast<uint16_t>(), outLocal.template ReinterpretCast<uint16_t>(), computeNum * 2);
        PipeBarrier<PIPE_ALL>();
        Select( mergeLocal.template ReinterpretCast<float>(), mblcmpLocal, tempLocal0.template ReinterpretCast<float>(), 
                mergeLocal.template ReinterpretCast<float>(), SELMODE::VSEL_TENSOR_TENSOR_MODE, computeNum);
        PipeBarrier<PIPE_ALL>();
       
        // 从mbl_inLocal获取提取码字掩码和右移的除数
        Broadcast<float, 2, 1>(tempLocal1.template ReinterpretCast<float>(), mbl_inLocal.template ReinterpretCast<float>(), dstShape_0, srcShape_0);
        PipeBarrier<PIPE_ALL>();

        // 得到提取码字掩码
        ShiftRight(tempLocal0, tempLocal1, (uint32_t)14, computeNum);
        PipeBarrier<PIPE_ALL>();

        // 从更新后的mergelocal获取数据
        And(outLocal.template ReinterpretCast<uint16_t>(), mergeLocal.template ReinterpretCast<uint16_t>(), tempLocal0.template ReinterpretCast<uint16_t>(), computeNum * 2);
        PipeBarrier<PIPE_ALL>();
        // 通过gather解码表恢复数据
        ShiftLeft(outLocal, outLocal, (uint32_t)2, computeNum);
        PipeBarrier<PIPE_ALL>();
        Gather(outLocal, tableLocal, outLocal, (uint32_t)0, (uint32_t)computeNum);
        PipeBarrier<PIPE_ALL>();
        // 至此指数部分已经完全复原

        // 更新mergeLocal，使用除法实现不同通道的位移
        ShiftLeft(tempLocal0, tempLocal1, (uint32_t)18, computeNum);
        PipeBarrier<PIPE_ALL>();
        ShiftRight(tempLocal0, tempLocal0, (uint32_t)23, computeNum);
        PipeBarrier<PIPE_ALL>();

        // int32_t eventIDVToV = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_V));
        // SetFlag<HardEvent::V_V>(eventIDVToV);
        // WaitFlag<HardEvent::V_V>(eventIDVToV);
        // DataCopy(outLocal, tempLocal0[computeNum / 2], 512);
        // PipeBarrier<PIPE_ALL>();
        Div(mergeLocal.template ReinterpretCast<float>(), mergeLocal.template ReinterpretCast<float>(), tempLocal0.template ReinterpretCast<float>(), (int32_t)computeNum);
        PipeBarrier<PIPE_ALL>();
        Cast(mergeLocal.template ReinterpretCast<int32_t>(), mergeLocal.template ReinterpretCast<float>(), RoundMode::CAST_TRUNC, computeNum);
        PipeBarrier<PIPE_ALL>();

        // 开始sign+指数+尾数的组合
        ShiftRight((ms_inLocal.template ReinterpretCast<uint16_t>())[computeNum / 2], ms_inLocal.template ReinterpretCast<uint16_t>(), (uint16_t)8, computeNum / 2);
        PipeBarrier<PIPE_ALL>();
        ShiftLeft(ms_inLocal.template ReinterpretCast<uint16_t>(), ms_inLocal.template ReinterpretCast<uint16_t>(), (uint16_t)8, computeNum / 2);
        PipeBarrier<PIPE_ALL>();
        ShiftRight(ms_inLocal.template ReinterpretCast<uint16_t>(), ms_inLocal.template ReinterpretCast<uint16_t>(), (uint16_t)8, computeNum / 2);
        PipeBarrier<PIPE_ALL>();
        // 尾数+sign展开操作完成

        ShiftLeft(outLocal, outLocal, (uint32_t)8, computeNum / 2);
        PipeBarrier<PIPE_ALL>();
        ShiftLeft(outLocal[computeNum / 2], outLocal[computeNum / 2], (uint32_t)24, computeNum / 2);
        PipeBarrier<PIPE_ALL>();
        Or(outLocal.template ReinterpretCast<uint16_t>(), outLocal.template ReinterpretCast<uint16_t>(), outLocal[computeNum / 2].template ReinterpretCast<uint16_t>(), computeNum / 2 * 2);
        PipeBarrier<PIPE_ALL>();
        // 指数位置已经就位

        Or(outLocal.template ReinterpretCast<uint16_t>(), outLocal.template ReinterpretCast<uint16_t>(), ms_inLocal.template ReinterpretCast<uint16_t>(), computeNum * 2);
        PipeBarrier<PIPE_ALL>();
        // 组合完成

        ShiftLeft(tempLocal0.template ReinterpretCast<uint16_t>(), outLocal.template ReinterpretCast<uint16_t>(), (uint16_t)15, computeNum);
        PipeBarrier<PIPE_ALL>();
        ShiftRight(outLocal.template ReinterpretCast<uint16_t>(), outLocal.template ReinterpretCast<uint16_t>(), (uint16_t)1, computeNum);
        PipeBarrier<PIPE_ALL>();
        Or(outLocal.template ReinterpretCast<uint16_t>(), outLocal.template ReinterpretCast<uint16_t>(), tempLocal0.template ReinterpretCast<uint16_t>(), computeNum );
        PipeBarrier<PIPE_ALL>();
        // 恢复数据
        // if(i == 505){
        // if(i == 512){
        //     DumpTensor(outLocal, 1, 4096);
        // }

        outQueue.EnQue(outLocal);
        ms_inQueue.FreeTensor(ms_inLocal);
        mbl_inQueue.FreeTensor(mbl_inLocal);
    }

    __aicore__ inline void ComputeFirst(
                                    uint32_t& accCompressed,       
                                    LocalTensor<T>& e_inLocal,// 32 + 32kb
                                    LocalTensor<T>& cmblLocal,// 2kb
                                    LocalTensor<T>& mergeLocal,// 32kb
                                    LocalTensor<T>& mblcmpLocal,// 2kb
                                    LocalTensor<T>& takeLocal,// 2kb
                                    LocalTensor<T>& tableLocal,// 2kb
                                    LocalTensor<T>& tempLocal0,// 32kb
                                    LocalTensor<T>& tempLocal1,// 32kb
                                    LocalTensor<T>& tempLocal2,// 2kb
                                    LocalTensor<T>& tempLocal3,// 2kb
                                    LocalTensor<T>& offsetLocal,// 2kb
                                    LocalTensor<T>& mask15Local// 2kb
    ){
        LocalTensor<T> ms_inLocal = ms_inQueue.DeQue<T>();// 16kb
        LocalTensor<T> outLocal = outQueue.AllocTensor<T>();// 32kb
        //必须注意：当and和or的操作类型为uint32_t，需要将操作数量翻倍
        // 每个块的第一次的mbl都没保存，被cmbl最终替代，第一次的mergeLocal就是第一次的数据

        // 通过gather解码表恢复数据
        ShiftLeft(outLocal, mergeLocal, (uint32_t)2, computeNum);
        PipeBarrier<PIPE_ALL>();
        Gather(outLocal, tableLocal, outLocal, (uint32_t)0, (uint32_t)computeNum);
        PipeBarrier<PIPE_ALL>();
        // 至此指数部分已经完全复原

        // 开始sign+指数+尾数的组合
        ShiftRight((ms_inLocal.template ReinterpretCast<uint16_t>())[computeNum / 2], ms_inLocal.template ReinterpretCast<uint16_t>(), (uint16_t)8, computeNum / 2);
        ShiftLeft(ms_inLocal.template ReinterpretCast<uint16_t>(), ms_inLocal.template ReinterpretCast<uint16_t>(), (uint16_t)8, computeNum / 2);
        ShiftRight(ms_inLocal.template ReinterpretCast<uint16_t>(), ms_inLocal.template ReinterpretCast<uint16_t>(), (uint16_t)8, computeNum / 2);
        // 尾数+sign展开操作完成

        ShiftLeft(outLocal, outLocal, (uint32_t)8, computeNum / 2);
        PipeBarrier<PIPE_ALL>();
        ShiftLeft(outLocal[computeNum / 2], outLocal[computeNum / 2], (uint32_t)24, computeNum / 2);
        PipeBarrier<PIPE_ALL>();
        Or(outLocal.template ReinterpretCast<uint16_t>(), outLocal.template ReinterpretCast<uint16_t>(), outLocal[computeNum / 2].template ReinterpretCast<uint16_t>(), computeNum / 2 * 2);
        PipeBarrier<PIPE_ALL>();
        // 指数位置已经就位

        Or(outLocal.template ReinterpretCast<uint16_t>(), outLocal.template ReinterpretCast<uint16_t>(), ms_inLocal.template ReinterpretCast<uint16_t>(), computeNum * 2);
        PipeBarrier<PIPE_ALL>();
        // 组合完成

        ShiftLeft(tempLocal0.template ReinterpretCast<uint16_t>(), outLocal.template ReinterpretCast<uint16_t>(), (uint16_t)15, computeNum);
        PipeBarrier<PIPE_ALL>();
        ShiftRight(outLocal.template ReinterpretCast<uint16_t>(), outLocal.template ReinterpretCast<uint16_t>(), (uint16_t)1, computeNum);
        PipeBarrier<PIPE_ALL>();
        Or(outLocal.template ReinterpretCast<uint16_t>(), outLocal.template ReinterpretCast<uint16_t>(), tempLocal0.template ReinterpretCast<uint16_t>(), computeNum );
        PipeBarrier<PIPE_ALL>();
        // 恢复数据

        outQueue.EnQue(outLocal);
        ms_inQueue.FreeTensor(ms_inLocal);
    }

    __aicore__ inline void CopyOut(uint32_t datablockId) {
        LocalTensor<T> outLocal = outQueue.DeQue<T>();
        DataCopy(output[datablockId * (datablockSize / sizeof(T))], outLocal, datablockSize / sizeof(T));
        outQueue.FreeTensor(outLocal);
    }

private:
    TPipe* pipe;

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

    uint32_t srcShape_0[2];
    uint32_t dstShape_0[2];
    uint32_t srcShape_1[2];
    uint32_t dstShape_1[2];
};

__global__ __aicore__ void compBF16(
                                uint32_t BUFFER_NUM,
                                uint32_t elementNum,
                                uint32_t tileLength,
                                uint32_t tileNum,
                                uint32_t threadblockNum,
                                uint32_t datablockNum,
                                uint32_t datablockSize,
                                __gm__ uint8_t* eGlobal, //e_input
                                __gm__ uint8_t* tableGlobal,
                                __gm__ uint8_t* msGlobal,
                                __gm__ uint8_t* mblGlobal,
                                __gm__ uint8_t* compSizePrefix,
                                __gm__ uint8_t* decompressedGlobal
                                ){
    TPipe pipe;
    DecompressKernelBF16<uint32_t> op;
    // assert(BUFFER_NUM == 1);
    op.Init(&pipe, BUFFER_NUM, elementNum, tileLength, tileNum, threadblockNum, datablockNum, datablockSize,
            eGlobal, tableGlobal, msGlobal, mblGlobal, compSizePrefix, decompressedGlobal);
    op.Process();
}

extern "C" void decompress(Header* cphd, void* stream, uint8_t* compressed, uint8_t* decompressed) {
    switch(cphd->dataType) {
        case 0:{ // BF16
            uint32_t elementNum = cphd->dataBlockSize / sizeof(uint16_t);
            // assert(elementNum == 8192);
            uint32_t tileNum = elementNum / cphd->tileLength;
            compBF16<<<
            // 2
            cphd->threadBlockNum
            , nullptr, stream>>>(1, elementNum, cphd->tileLength, tileNum, cphd->threadBlockNum, cphd->dataBlockNum, cphd->dataBlockSize,
                                    getCompressed_exp(cphd, compressed), getTable(cphd, compressed), getMsdata(cphd, compressed), getMbl(cphd, compressed), getCompSizePrefix(cphd, compressed), decompressed);
            break;
        }
        case 1:{ // FP16
            
            break;
        }
        case 2:{ // FP32

            break;
        }
        default:{
            // printf("Unsupported data type: %u\n", dataType);
            return;
        }

    }
}


