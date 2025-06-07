#include <chrono>
#include "kernel_operator.h"

using namespace AscendC;

constexpr uint32_t DATA_BLOCK_BYTE_NUM = 4096;// 单位为字节
constexpr int32_t BUFFER_NUM = 2; // 双缓冲
constexpr int32_t BLOCK_NUM = 256;// block的数量
constexpr uint32_t HISTOGRAM_BINS = 256;// 尽可能是2的幂，直方图桶数
constexpr uint32_t TILE_LEN = 16; // 每个Tile处理16个单元(单元指输入数据的类型)
constexpr uint32_t TILE_NUM = DATA_BLOCK_BYTE_NUM / sizeof(uint32_t) / TILE_LEN; // 每个数据块包含TILE_NUM个TILE

constexpr uint32_t NUM = 64;
constexpr uint32_t TILE_LEN_C = TILE_LEN * NUM * 2;
constexpr uint32_t TILE_NUM_C = DATA_BLOCK_BYTE_NUM / sizeof(uint32_t) * 2 / TILE_LEN_C;

//注意：所有算子的输入与输出尽可能32字节对齐，Add这些底层接口的输入与输出必须32字节对齐
/*
template<typename T>//T = uint32_t
class Extractbits_and_histogramKernel {
public:
    __aicore__ inline Extractbits_and_histogramKernel() {} // 切分数据，分离指数位，同时进行histogram统计
    // 输入：uint16_t数组(两两组成一个int32_t)
    // 输出：指数数组（int32_t），尾数+sign数组（uint8_t)，histogram统计数组（int32_t）

    __aicore__ inline void Init(TPipe* pipe,
                                uint32_t datablockNum,
                                __gm__ uint8_t* in, 
                                __gm__ uint8_t* tempBuffer, 
                                __gm__ uint8_t* final, 
                                __gm__ uint8_t* histogramDevice, 
                                uint32_t totalUncompressedSize) {
        this->pipe = pipe;
        
        this->blockId = GetBlockIdx();
        this->blockNum = GetBlockNum();
        this->datablockNum = datablockNum;

        input.SetGlobalBuffer((__gm__ T*)(in));
        e_output.SetGlobalBuffer((__gm__ T*)(tempBuffer));
        m_s_output.SetGlobalBuffer((__gm__ T*)(final + 32 + HISTOGRAM_BINS * sizeof(uint8_t) + TILE_NUM * datablockNum));// 32字节对齐
        hist_output.SetGlobalBuffer((__gm__ int32_t*)(histogramDevice + sizeof(int32_t) * HISTOGRAM_BINS * blockId));

        pipe->InitBuffer(inQueue, BUFFER_NUM, TILE_LEN * sizeof(T));
        pipe->InitBuffer(e_outQueue, BUFFER_NUM, TILE_LEN * sizeof(T));
        pipe->InitBuffer(m_s_outQueue, BUFFER_NUM, TILE_LEN / 2 * sizeof(T));
        //因为开启了double_buffer，最多只能开四个queue
    }

    __aicore__ inline void Process() {

        pipe->InitBuffer(calcBuf0, TILE_LEN * sizeof(T));
        pipe->InitBuffer(calcBuf1, TILE_LEN * sizeof(T));
        pipe->InitBuffer(calcBuf2, TILE_LEN * sizeof(T));
        pipe->InitBuffer(calcBuf3, TILE_LEN * sizeof(T));
        pipe->InitBuffer(tempHist, TILE_LEN * HISTOGRAM_BINS * sizeof(int32_t));
        pipe->InitBuffer(histBuffer0, TILE_LEN * sizeof(int32_t));
        pipe->InitBuffer(histBuffer1, TILE_LEN * sizeof(int32_t));
        pipe->InitBuffer(mask0, TILE_LEN * sizeof(T));
        pipe->InitBuffer(mask1, TILE_LEN * sizeof(T));
        pipe->InitBuffer(mask2, TILE_LEN * sizeof(T));
        pipe->InitBuffer(mask3, TILE_LEN * sizeof(T));
        pipe->InitBuffer(mask4, TILE_LEN * sizeof(T));
        pipe->InitBuffer(offsetBuffer, TILE_LEN * sizeof(int32_t));
        pipe->InitBuffer(one, TILE_LEN * sizeof(int32_t));

        LocalTensor<T> tempLocal0 = calcBuf0.Get<T>();
        LocalTensor<T> tempLocal1 = calcBuf1.Get<T>();
        LocalTensor<T> tempLocal2 = calcBuf2.Get<T>();
        LocalTensor<T> tempLocal3 = calcBuf3.Get<T>();
        LocalTensor<int32_t> histogram = tempHist.Get<int32_t>();
        LocalTensor<int32_t> histTensor0 = histBuffer0.Get<int32_t>();
        LocalTensor<int32_t> histTensor1 = histBuffer1.Get<int32_t>();
        LocalTensor<T> mask0_tensor = mask0.Get<T>();
        LocalTensor<T> mask1_tensor = mask1.Get<T>();
        LocalTensor<T> mask2_tensor = mask2.Get<T>();
        LocalTensor<T> mask3_tensor = mask3.Get<T>();
        LocalTensor<T> mask4_tensor = mask4.Get<T>();
        LocalTensor<int32_t> all_one = one.Get<int32_t>();
        LocalTensor<int32_t> offset_tensor = offsetBuffer.Get<int32_t>();

        Duplicate(histogram, (int32_t)0, HISTOGRAM_BINS * TILE_LEN);// 初始化全0
        Duplicate(mask0_tensor, (T)65280, TILE_LEN);//11111111 00000000
        Duplicate(mask1_tensor, (T)255, TILE_LEN);//00000000 11111111
        Duplicate(mask2_tensor, (T)16711935, TILE_LEN);//00000000 11111111 00000000 11111111
        Duplicate(mask3_tensor, (T)4278255360, TILE_LEN);//11111111 00000000 11111111 00000000
        Duplicate(mask4_tensor, (T)65535, TILE_LEN);//00000000 00000000 11111111 11111111
        Duplicate(all_one, (int32_t)1, TILE_LEN);//0000 0000 0000 0001
        uint32_t num = ((1 << 16) + 1) * HISTOGRAM_BINS * sizeof(T);
        for(int i = 0; i < TILE_LEN; i ++){
            offset_tensor(i) = i * num;
        }
        // printf("datablockNum: %d\n", datablockNum);
        for(int i = blockId; i < datablockNum; i += blockNum){
            // if(blockId == 1)
            // {
            //     printf("i:%d\n", i);
            // }
            int offset0 = i * DATA_BLOCK_BYTE_NUM / sizeof(T);// 指数原大小保存
            for(int tileIdx = 0; tileIdx < TILE_NUM; tileIdx ++){
                int offset1 = tileIdx * TILE_LEN;
                int offset = offset0 + offset1;
                CopyIn(offset);
                Compute(mask0_tensor, mask1_tensor, mask2_tensor, mask3_tensor, mask4_tensor, all_one, offset_tensor, histogram, tempLocal0, tempLocal1, tempLocal2, tempLocal3, histTensor0, histTensor1);
                CopyOut(offset);
            }
        }
        MergeLocalHist(histogram);// 合并TILE_LEN个temp直方图为最终的一个
    }

private:
    __aicore__ inline void CopyIn(int32_t offset) {
        LocalTensor<T> inLocal = inQueue.AllocTensor<T>();

        DataCopy(inLocal, input[offset], TILE_LEN);
        inQueue.EnQue(inLocal);
    }

    __aicore__ inline void Compute(LocalTensor<T>& mask0_tensor,
                                   LocalTensor<T>& mask1_tensor,
                                   LocalTensor<T>& mask2_tensor,
                                   LocalTensor<T>& mask3_tensor,
                                   LocalTensor<T>& mask4_tensor,
                                   LocalTensor<int32_t>& all_one,
                                   LocalTensor<int32_t>& offset_tensor,
                                   LocalTensor<int32_t>& histogram,
                                   LocalTensor<T>& tempLocal0,
                                   LocalTensor<T>& tempLocal1,
                                   LocalTensor<T>& tempLocal2,
                                   LocalTensor<T>& tempLocal3,
                                   LocalTensor<int32_t>& histTensor0,
                                   LocalTensor<int32_t>& histTensor1
                                   ) {
        LocalTensor<T> inLocal = inQueue.DeQue<T>();
        LocalTensor<T> e_outLocal = e_outQueue.AllocTensor<T>();
        LocalTensor<T> m_s_outLocal = m_s_outQueue.AllocTensor<T>();

        // len /= 2;
        // 处理每个元素，每次提取32个int32_t（取出64个uint16_t）
        ShiftLeft(tempLocal0, inLocal, (uint32_t)1, TILE_LEN);
        ShiftRight(tempLocal1, inLocal, (uint32_t)31, TILE_LEN);//int类型自动算数移位,uint32_t为逻辑移位
        Or(tempLocal2, tempLocal0, tempLocal1, (int32_t)TILE_LEN * 2);//将sign放在最后

        And(tempLocal0, tempLocal2, mask2_tensor, (uint32_t)TILE_LEN * 2);//取出从高到低1和3字节，尾数部分
        ShiftLeft(tempLocal1, tempLocal0[TILE_LEN / 2], (uint32_t)8, (uint32_t)(TILE_LEN / 2));
        Or(m_s_outLocal, tempLocal0, tempLocal1, (int32_t)TILE_LEN);// 对半折叠存储

        And(tempLocal3, tempLocal2, mask3_tensor, (int32_t)TILE_LEN * 2);//取出从高到低0和2字节，指数部分
        ShiftRight(e_outLocal, tempLocal3, (uint32_t)8, (uint32_t)TILE_LEN);//右移8位
        ShiftRight(tempLocal1, tempLocal3, (uint32_t)(8 - 2), (int32_t)TILE_LEN);// 因为是uint32_t，需要乘四字节，所以少右移2位
        Add(tempLocal2.template ReinterpretCast<int32_t>(), tempLocal1.template ReinterpretCast<int32_t>(), offset_tensor, (int32_t)TILE_LEN);

        And(tempLocal0, tempLocal2, mask4_tensor, (int32_t)TILE_LEN * 2);//取出低16位
        Gather(histTensor0, histogram, tempLocal0, (uint32_t)0, (uint32_t)TILE_LEN);// offset为字节单位
        Add(histTensor1, histTensor0, all_one, (int32_t)TILE_LEN);
        for(int i = 0; i < TILE_LEN; i ++){
            histogram(tempLocal0(i) / sizeof(T)) = histTensor1(i);//需要除sizeof(T)转成T为单位
        }
        // Scatter(histogram.template ReinterpretCast<uint32_t>(), histTensor1.template ReinterpretCast<uint32_t>(), tempLocal0, (uint32_t)0, (uint32_t)len);

        ShiftRight(tempLocal0, tempLocal2, (uint32_t)16, (int32_t)TILE_LEN);//取出高16位
        Gather(histTensor0, histogram, tempLocal0, (uint32_t)0, (uint32_t)TILE_LEN);
        Add(histTensor1, histTensor0, all_one, (int32_t)TILE_LEN);
        for(int i = 0; i < TILE_LEN; i ++){
            histogram(tempLocal0(i) / sizeof(T)) = histTensor1(i);
        }
        // Scatter(histogram, histTensor1, tempLocal0, (uint32_t)0, (uint32_t)len);

        inQueue.FreeTensor(inLocal);
        e_outQueue.EnQue(e_outLocal);
        m_s_outQueue.EnQue(m_s_outLocal);
    }

    __aicore__ inline void CopyOut(int32_t offset) {
        LocalTensor<T> e_outLocal = e_outQueue.DeQue<T>();
        LocalTensor<T> m_s_outLocal = m_s_outQueue.DeQue<T>();

        DataCopy(e_output[offset], e_outLocal, TILE_LEN);
        DataCopy(m_s_output[offset / 2], m_s_outLocal, TILE_LEN / 2);// 对半折叠

        e_outQueue.FreeTensor(e_outLocal);
        m_s_outQueue.FreeTensor(m_s_outLocal);
    }

    __aicore__ inline void MergeLocalHist(LocalTensor<int32_t>& histogram) {
        // for(int i = 1; i < TILE_NUM; i ++){
        //     Add(histogram, histogram, histogram[i * HISTOGRAM_BINS], (int32_t)HISTOGRAM_BINS);
        // }
        for(int i = 1; i < TILE_LEN; i ++){
            for(int j = 0; j < HISTOGRAM_BINS; j ++){
                histogram(j) = histogram(j) + histogram(i * HISTOGRAM_BINS + j);
            }
        }
        int sum = 0;
        for(int i = 0; i < HISTOGRAM_BINS; i ++)
            sum = sum + histogram(i);
        // if(blockId == 0) assert(sum == 2048);
        DataCopy(hist_output, histogram, HISTOGRAM_BINS);
    }

private:
    TPipe* pipe;
    TQue<QuePosition::VECIN, 1> inQueue;// 1代表队列的深度
    TQue<QuePosition::VECOUT, 1> e_outQueue;
    TQue<QuePosition::VECOUT, 1> m_s_outQueue;

    TBuf<TPosition::VECCALC> calcBuf0;
    TBuf<TPosition::VECCALC> calcBuf1;
    TBuf<TPosition::VECCALC> calcBuf2;
    TBuf<TPosition::VECCALC> calcBuf3;
    TBuf<TPosition::VECCALC> tempHist;
    TBuf<TPosition::VECCALC> histBuffer0;
    TBuf<TPosition::VECCALC> histBuffer1;
    TBuf<TPosition::VECCALC> mask0;
    TBuf<TPosition::VECCALC> mask1;
    TBuf<TPosition::VECCALC> mask2;
    TBuf<TPosition::VECCALC> mask3;
    TBuf<TPosition::VECCALC> mask4;
    TBuf<TPosition::VECCALC> one;
    TBuf<TPosition::VECCALC> offsetBuffer;

    GlobalTensor<T> input;
    GlobalTensor<T> e_output;
    GlobalTensor<T> m_s_output;
    GlobalTensor<int32_t> hist_output;

    uint32_t blockId;
    uint32_t blockNum;
    uint32_t datablockNum;
};

template<typename T>// int32_t
class MergeHistogramKernel {
public:
    __aicore__ inline MergeHistogramKernel() {} // 合并blockNum个直方图，生成全局直方图和全局编码表

    __aicore__ inline void Init(TPipe* pipe,
                                __gm__ uint8_t* hist_in,
                                __gm__ uint8_t* final_table) {
        this->pipe = pipe;
        this->blockId = GetBlockIdx();
        this->blockNum = GetBlockNum();

        hist.SetGlobalBuffer((__gm__ T*)(hist_in));
        table.SetGlobalBuffer((__gm__ uint8_t*)(final_table));

        pipe->InitBuffer(inQueue, BUFFER_NUM, HISTOGRAM_BINS * sizeof(T));
    }

    __aicore__ inline void Process() {
        pipe->InitBuffer(temp, HISTOGRAM_BINS * sizeof(T));
        LocalTensor<T> tempLocal = temp.Get<T>();
        Duplicate(tempLocal, (int32_t)0, HISTOGRAM_BINS);

        pipe->InitBuffer(sorttemp, HISTOGRAM_BINS * sizeof(uint64_t));
        LocalTensor<uint64_t> sortLocal = sorttemp.Get<uint64_t>();

        pipe->InitBuffer(tabletemp, HISTOGRAM_BINS * sizeof(uint8_t));
        LocalTensor<uint8_t> tableLocal = tabletemp.Get<uint8_t>();
        assert(tempLocal(0) == 0);
        for(int i = 0; i < //2
        BLOCK_NUM
        ; i ++){
            CopyIn(i);
            Compute(tempLocal);
        }

        Sort(tempLocal, sortLocal);
        Generate_table(tempLocal, sortLocal, tableLocal);
        // assert(tempLocal(0) == 0);
        DataCopy(hist, tempLocal, HISTOGRAM_BINS);
        DataCopy(table, tableLocal, HISTOGRAM_BINS);
    }

private:
    __aicore__ inline void CopyIn(uint32_t offset) {
        LocalTensor<T> inLocal = inQueue.AllocTensor<T>();
        DataCopy(inLocal, hist[offset * HISTOGRAM_BINS], HISTOGRAM_BINS);
        // if(offset == 1){
            // assert(inLocal(117) == 1413);
            // assert(inLocal(119) == 0);
        // }
        assert(inLocal(0) == 0);
        inQueue.EnQue(inLocal);
    }

    __aicore__ inline void Compute(LocalTensor<T>& tempLocal){
        LocalTensor<T> inLocal = inQueue.DeQue<T>();
        Add(tempLocal, inLocal, tempLocal, (T)HISTOGRAM_BINS);
        inQueue.FreeTensor(inLocal);
    }

    __aicore__ inline void Sort(LocalTensor<T>& tempLocal, LocalTensor<uint64_t> sortLocal){
        for(int i = 0; i < HISTOGRAM_BINS; i ++){
            sortLocal(i) = (((uint64_t)tempLocal(i)) << 32) | i;
        }
        for (int i = 0; i < HISTOGRAM_BINS - 1; i++) {
            for (int j = 0; j < HISTOGRAM_BINS - i - 1; j++) {
                if (sortLocal(j) < sortLocal(j + 1)) {
                    uint64_t temp = sortLocal(j);
                    sortLocal(j) = sortLocal(j + 1);
                    sortLocal(j + 1) = temp;
                }
            }
        }
    }
    //  如果两个数相同，序号大的在前
    __aicore__ inline void Generate_table(LocalTensor<T>& tempLocal, LocalTensor<uint64_t>& sortLocal, LocalTensor<uint8_t>& tableLocal){
        for(int i = 0; i < HISTOGRAM_BINS; i ++){
            tempLocal(sortLocal(i) & 0xffffffff) = i;
        }
        for (int i = 0; i < HISTOGRAM_BINS; i++) {
            tableLocal(i) = (uint8_t)tempLocal(i);
        }
    }

private:
    TPipe* pipe;
    TQue<QuePosition::VECIN, 1> inQueue;
    TBuf<QuePosition::VECCALC> temp;
    TBuf<QuePosition::VECCALC> sorttemp;
    TBuf<QuePosition::VECCALC> tabletemp;

    GlobalTensor<T> hist;
    GlobalTensor<uint8_t> table;

    uint32_t blockId;
    uint32_t blockNum;
};
*/

template<typename T>// uint32_t
class CompressKernel {
public:
    __aicore__ inline CompressKernel() {}
    // 输入：指数数组（uint8_t），32位编码表，
    // 输出：max_bit_length数组（4bits * tile_num * 2），码字（max_bit_length * tile_len,最终的压缩块大小需要向上取到32的倍数，满足32字节对齐的要求）

    __aicore__ inline void Init(TPipe* pipe,
                                uint32_t datablock,
                                __gm__ uint8_t* tempBuffer, //e_input
                                __gm__ uint8_t* final, //output
                                __gm__ uint8_t* histogramDevice, //table_input
                                __gm__ uint8_t* tilePrefix,// 每个数据block中每个tile之后压缩大小
                                uint32_t totalUncompressedBytes // 保存全部未压缩数据的大小，用于分块拖尾处理
                                ) {
        this->pipe = pipe;
        this->blockId = GetBlockIdx(); //获取当前blockId
        this->blockNum = GetBlockNum(); //获取当前blockNum
        this->totalUncompressedBytes = totalUncompressedBytes;
        this->datablockNum = datablock;

        int datablockNumPerBLOCK = (datablockNum + BLOCK_NUM - 1) / BLOCK_NUM;

        srcShape_[0] = TILE_LEN_C / TILE_LEN;
        srcShape_[1] = 1;
        dstShape_[0] = TILE_LEN_C / TILE_LEN;
        dstShape_[1] = TILE_LEN;

        e_input.SetGlobalBuffer((__gm__ T*)(tempBuffer));
        table_input.SetGlobalBuffer((__gm__ T*)(histogramDevice));
        mbl_output.SetGlobalBuffer((__gm__ uint8_t*)(final + 32 + HISTOGRAM_BINS));
        output.SetGlobalBuffer((__gm__ T*)(final + 32 + HISTOGRAM_BINS + TILE_NUM * datablockNum + DATA_BLOCK_BYTE_NUM / 2 * datablockNum + (DATA_BLOCK_BYTE_NUM / 2 * datablockNumPerBLOCK) * BLOCK_NUM + (DATA_BLOCK_BYTE_NUM / 2 * datablockNumPerBLOCK) * blockId));

        pipe->InitBuffer(inQueue, BUFFER_NUM, TILE_LEN_C * sizeof(T));// 初始化Pipe缓冲区，每个Tile大小TILE_LEN
        pipe->InitBuffer(e_outQueue, BUFFER_NUM, TILE_LEN_C * sizeof(uint16_t));// 字节为单位
        pipe->InitBuffer(mbl_outQueue, BUFFER_NUM, TILE_NUM * 2);// 初始化Pipe缓冲区，每个Tile大小TILE_LEN
    }

    __aicore__ inline void Process(//uint32_t& compressedSize
                                    ) {
        pipe->InitBuffer(temp, TILE_LEN_C * sizeof(T));
        pipe->InitBuffer(encoded, TILE_LEN_C * sizeof(T));
        pipe->InitBuffer(table, HISTOGRAM_BINS * sizeof(T));
        pipe->InitBuffer(bltable, HISTOGRAM_BINS * sizeof(T));
        pipe->InitBuffer(bl, TILE_LEN_C * sizeof(T));
        pipe->InitBuffer(mbl, NUM * 2 * sizeof(T));
        pipe->InitBuffer(mblbroadcast, TILE_LEN_C * sizeof(T));
        pipe->InitBuffer(power, HISTOGRAM_BINS * sizeof(T));
        pipe->InitBuffer(multiplier, TILE_LEN_C * sizeof(T));
        pipe->InitBuffer(merge, TILE_LEN_C * sizeof(T));
        pipe->InitBuffer(cmbl, TILE_LEN_C * sizeof(T));
        pipe->InitBuffer(mask, TILE_LEN_C / sizeof(uint8_t) * sizeof(T));
        pipe->InitBuffer(normalization, TILE_LEN_C * sizeof(T));
        pipe->InitBuffer(gather, TILE_LEN_C * sizeof(T));
        pipe->InitBuffer(clear, TILE_LEN_C * sizeof(T));
        
        LocalTensor<T> tempLocal = temp.Get<T>();
        LocalTensor<T> encodedLocal = encoded.Get<T>();
        LocalTensor<T> tableLocal = table.Get<T>();
        LocalTensor<T> bltableLocal = bltable.Get<T>();
        LocalTensor<T> blLocal = bl.Get<T>();
        LocalTensor<T> mblLocal = mbl.Get<T>();
        LocalTensor<T> mblbroadcastLocal = mblbroadcast.Get<T>();
        LocalTensor<T> powerLocal = power.Get<T>();
        LocalTensor<T> multiplierLocal = multiplier.Get<T>();
        LocalTensor<T> mergeLocal = merge.Get<T>();
        LocalTensor<T> cmblLocal = cmbl.Get<T>();
        LocalTensor<T> maskLocal = mask.Get<T>();
        LocalTensor<T> normalizationLocal = normalization.Get<T>();
        LocalTensor<T> gatherLocal = gather.Get<T>();
        LocalTensor<T> clearLocal = clear.Get<T>();

        DataCopy(tableLocal, table_input, HISTOGRAM_BINS);

        uint32_t j = 0;
        uint32_t start = 0;
        for(int i = 1; i <= HISTOGRAM_BINS; i <<= 1){
            for(int k = start; k < i; k ++){
                bltableLocal(k) = j;
            }
            start = i;
            j ++;
        }

        Duplicate(normalizationLocal, (T)15, TILE_LEN_C);// 00000000 00000000 00000000 00001111
        Duplicate(clearLocal, (T)65535, TILE_LEN_C);// 00000000 00000000 11111111 11111111

        uint32_t compressedSize = 0;// 字节为单位
        uint32_t totalcompressedSize = 0;// 字节为单位
        for(uint32_t i = blockId; i < datablockNum; i += blockNum){
            uint32_t offset = i * TILE_LEN_C;
            CopyIn(offset);
            Compute(
                compressedSize, 
                tempLocal, encodedLocal, tableLocal, bltableLocal, blLocal, mblLocal, mblbroadcastLocal, powerLocal, multiplierLocal, 
                    mergeLocal, cmblLocal, maskLocal, normalizationLocal, gatherLocal, clearLocal);
            CopyOut(totalcompressedSize, compressedSize, i);// 拷贝结果到GM，这里拷贝到字节数必定为32字节的倍数
            totalcompressedSize = totalcompressedSize + (uint32_t)compressedSize;// 更新totalcompressedSize
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t offset) {
        LocalTensor<T> inLocal = inQueue.AllocTensor<T>();
        // 拷贝当前Tile数据到Local
        DataCopy(inLocal, e_input[offset], TILE_LEN_C);
        inQueue.EnQue(inLocal);
    }

    __aicore__ inline void Compute(uint32_t& thiscompressedSize,         
                                   LocalTensor<T>& tempLocal,// TILE_LEN_C
                                   LocalTensor<T>& encodedLocal,
                                   LocalTensor<T>& tableLocal,// HISTOGRAM_BINS
                                   LocalTensor<T>& bltableLocal,// HISTOGRAM_BINS
                                   LocalTensor<T>& blLocal,// TILE_LEN_C
                                   LocalTensor<T>& mblLocal,// NUM * 2
                                   LocalTensor<T>& mblbroadcastLocal,// TILE_LEN_C
                                   LocalTensor<T>& powerLocal,// HISTOGRAM_BINS
                                   LocalTensor<T>& multiplierLocal,// TILE_LEN_C
                                   LocalTensor<T>& mergeLocal,// TILE_LEN_C
                                   LocalTensor<T>& cmblLocal,// TILE_LEN_C
                                   LocalTensor<T>& maskLocal,// TILE_LEN_C / sizeof(uint8_t)
                                   LocalTensor<T>& normalizationLocal,// TILE_LEN_C
                                   LocalTensor<T>& gatherLocal,// TILE_LEN_C
                                   LocalTensor<T>& clearLocal// TILE_LEN_C
    ) {

        LocalTensor<T> e_inLocal = inQueue.DeQue<T>();// TILE_LEN_C
        LocalTensor<T> e_outLocal = e_outQueue.AllocTensor<T>();// TILE_LEN_C
        LocalTensor<uint8_t> mbl_outLocal = mbl_outQueue.AllocTensor<uint8_t>();// NUM * 2 / 8

        uint32_t compressedSize = 0;
        uint32_t cmbl = 0;

        for(int i = 0; i < TILE_NUM * 2; i ++){
            ShiftLeft(tempLocal, e_inLocal[i * TILE_LEN], (uint32_t)2, TILE_LEN);
            Gather(encodedLocal, tableLocal, tempLocal, (uint32_t)0, (uint32_t)TILE_LEN);

            ShiftLeft(tempLocal, encodedLocal, (uint32_t)2, TILE_LEN);
            Gather(blLocal, bltableLocal, tempLocal, (uint32_t)0, (uint32_t)TILE_LEN);

            ReduceMax<float>(mblLocal.template ReinterpretCast<float>(), blLocal.template ReinterpretCast<float>(), tempLocal.template ReinterpretCast<float>(), TILE_LEN, false);
            uint32_t mbl = mblLocal(0);
            mbl_outLocal(i) = (uint8_t)mbl;

            ShiftLeft(mergeLocal, mergeLocal, (uint32_t)mbl, TILE_LEN);
            Or(mergeLocal, mergeLocal, encodedLocal, TILE_LEN * 2);

            cmbl = cmbl + mbl;
            And(gatherLocal, mergeLocal, clearLocal, TILE_LEN * 2); 
            ShiftLeft(tempLocal, gatherLocal[TILE_LEN / 2], (uint32_t)16, TILE_LEN / 2);
            ShiftRight(mergeLocal, mergeLocal, (uint32_t)16, TILE_LEN);
            uint32_t NEW_TILE_LEN = (cmbl >> 4) * TILE_LEN;
            
            Or(e_outLocal[compressedSize / sizeof(T)], gatherLocal, tempLocal, NEW_TILE_LEN / 2 * 2);//必是32字节的倍数
            compressedSize = compressedSize + NEW_TILE_LEN * sizeof(uint16_t);//以字节为单位
            cmbl = cmbl & 15;
            
        }
        thiscompressedSize = compressedSize;
        mblLocal(0) = cmbl;
        ShiftLeft(tempLocal, mbl_outLocal[TILE_NUM].template ReinterpretCast<uint32_t>(), (uint32_t)4, TILE_NUM / sizeof(uint32_t));
        Or(mbl_outLocal.template ReinterpretCast<uint32_t>(), mbl_outLocal.template ReinterpretCast<uint32_t>(), tempLocal, TILE_NUM / sizeof(uint32_t) * 2);

        e_outQueue.EnQue(e_outLocal);
        mbl_outQueue.EnQue(mbl_outLocal);
        inQueue.FreeTensor(e_inLocal);
    }

    __aicore__ inline void CopyOut(uint32_t totalcompressedSize, uint32_t compressedSize, uint32_t datablockId) {
        LocalTensor<T> e_outLocal = e_outQueue.DeQue<T>();
        LocalTensor<uint8_t> mbl_outLocal = mbl_outQueue.DeQue<uint8_t>();
        // 拷贝当前压缩输出数据到GM 
        assert(compressedSize % 32 == 0);
        DataCopy(output[totalcompressedSize / sizeof(T)], e_outLocal, compressedSize / sizeof(T));
        // 拷贝当前mbl数据到GM
        DataCopy(mbl_output[datablockId * TILE_NUM], mbl_outLocal, TILE_NUM);
        e_outQueue.FreeTensor(e_outLocal);
        mbl_outQueue.FreeTensor(mbl_outLocal);
    }

private:
    TPipe* pipe;

    TQue<QuePosition::VECIN, 1> inQueue;
    TQue<QuePosition::VECOUT, 1> e_outQueue;
    TQue<QuePosition::VECOUT, 1> mbl_outQueue;

    TBuf<TPosition::VECCALC> temp;
    TBuf<TPosition::VECCALC> encoded;
    TBuf<TPosition::VECCALC> table;
    TBuf<TPosition::VECCALC> bltable;
    TBuf<TPosition::VECCALC> bl;
    TBuf<TPosition::VECCALC> mbl;
    TBuf<TPosition::VECCALC> mblbroadcast;
    TBuf<TPosition::VECCALC> power;
    TBuf<TPosition::VECCALC> multiplier;
    TBuf<TPosition::VECCALC> merge;
    TBuf<TPosition::VECCALC> cmbl;
    TBuf<TPosition::VECCALC> mask;
    TBuf<TPosition::VECCALC> normalization;
    TBuf<TPosition::VECCALC> gather;
    TBuf<TPosition::VECCALC> clear;

    GlobalTensor<T> e_input;
    GlobalTensor<T> table_input;
    GlobalTensor<uint8_t> mbl_output;
    GlobalTensor<T> output;

    uint32_t blockId;
    uint32_t blockNum;
    uint32_t totalUncompressedBytes;
    uint32_t datablockNum;

    uint32_t srcShape_[2];
    uint32_t dstShape_[2];
};

template<typename T>// uint32_t
class CompressKernel1 {
public:
    __aicore__ inline CompressKernel1() {}
    // 输入：指数数组（uint8_t），32位编码表，
    // 输出：max_bit_length数组（4bits * tile_num * 2），码字（max_bit_length * tile_len,最终的压缩块大小需要向上取到32的倍数，满足32字节对齐的要求）

    __aicore__ inline void Init(TPipe* pipe,
                                uint32_t datablock,
                                __gm__ uint8_t* tempBuffer, //e_input
                                __gm__ uint8_t* final, //output
                                __gm__ uint8_t* histogramDevice, //table_input
                                __gm__ uint8_t* tilePrefix,// 每个数据block中每个tile之后压缩大小
                                uint32_t totalUncompressedBytes // 保存全部未压缩数据的大小，用于分块拖尾处理
                                ) {
        this->pipe = pipe;
        this->blockId = GetBlockIdx(); //获取当前blockId
        this->blockNum = GetBlockNum(); //获取当前blockNum
        this->totalUncompressedBytes = totalUncompressedBytes;
        this->datablockNum = datablock;

        int datablockNumPerBLOCK = (datablockNum + BLOCK_NUM - 1) / BLOCK_NUM;

        srcShape_[0] = TILE_LEN_C / TILE_LEN;
        srcShape_[1] = 1;
        dstShape_[0] = TILE_LEN_C / TILE_LEN;
        dstShape_[1] = TILE_LEN;

        e_input.SetGlobalBuffer((__gm__ T*)(tempBuffer));
        table_input.SetGlobalBuffer((__gm__ T*)(histogramDevice));
        mbl_output.SetGlobalBuffer((__gm__ T*)(final + 32 + HISTOGRAM_BINS));
        output.SetGlobalBuffer((__gm__ T*)(final + 32 + HISTOGRAM_BINS + TILE_NUM * datablockNum + DATA_BLOCK_BYTE_NUM / 2 * datablockNum + (DATA_BLOCK_BYTE_NUM / 2 * datablockNumPerBLOCK) * BLOCK_NUM + (DATA_BLOCK_BYTE_NUM / 2 * datablockNumPerBLOCK) * blockId));

        pipe->InitBuffer(inQueue, BUFFER_NUM, TILE_LEN_C * sizeof(T));// 初始化Pipe缓冲区，每个Tile大小TILE_LEN
        pipe->InitBuffer(e_outQueue, BUFFER_NUM, TILE_LEN_C * sizeof(uint16_t));// 字节为单位
        pipe->InitBuffer(mbl_outQueue, BUFFER_NUM, NUM * 2 / 2);// 初始化Pipe缓冲区，每个Tile大小TILE_LEN
    }

    __aicore__ inline void Process(//uint32_t& compressedSize
                                    ) {
        pipe->InitBuffer(temp, TILE_LEN_C * sizeof(T));
        pipe->InitBuffer(encoded, TILE_LEN_C * sizeof(T));
        pipe->InitBuffer(table, HISTOGRAM_BINS * sizeof(T));
        pipe->InitBuffer(bltable, HISTOGRAM_BINS * sizeof(T));
        pipe->InitBuffer(bl, TILE_LEN_C * sizeof(T));
        pipe->InitBuffer(mbl, NUM * 2 * sizeof(T));
        pipe->InitBuffer(mblbroadcast, TILE_LEN_C * sizeof(T));
        pipe->InitBuffer(power, HISTOGRAM_BINS * sizeof(T));
        pipe->InitBuffer(multiplier, TILE_LEN_C * sizeof(T));
        pipe->InitBuffer(merge, TILE_LEN_C * sizeof(T));
        pipe->InitBuffer(cmbl, TILE_LEN_C * sizeof(T));
        pipe->InitBuffer(mask, TILE_LEN_C / sizeof(uint8_t) * sizeof(T));
        pipe->InitBuffer(normalization, TILE_LEN_C * sizeof(T));
        pipe->InitBuffer(gather, TILE_LEN_C * sizeof(T));
        pipe->InitBuffer(clear, TILE_LEN_C * sizeof(T));
        
        LocalTensor<T> tempLocal = temp.Get<T>();
        LocalTensor<T> encodedLocal = encoded.Get<T>();
        LocalTensor<T> tableLocal = table.Get<T>();
        LocalTensor<T> bltableLocal = bltable.Get<T>();
        LocalTensor<T> blLocal = bl.Get<T>();
        LocalTensor<T> mblLocal = mbl.Get<T>();
        LocalTensor<T> mblbroadcastLocal = mblbroadcast.Get<T>();
        LocalTensor<T> powerLocal = power.Get<T>();
        LocalTensor<T> multiplierLocal = multiplier.Get<T>();
        LocalTensor<T> mergeLocal = merge.Get<T>();
        LocalTensor<T> cmblLocal = cmbl.Get<T>();
        LocalTensor<T> maskLocal = mask.Get<T>();
        LocalTensor<T> normalizationLocal = normalization.Get<T>();
        LocalTensor<T> gatherLocal = gather.Get<T>();
        LocalTensor<T> clearLocal = clear.Get<T>();

        DataCopy(tableLocal, table_input, HISTOGRAM_BINS);

        uint32_t j = 0;
        uint32_t start = 0;
        for(int i = 1; i <= HISTOGRAM_BINS; i <<= 1){
            for(int k = start; k < i; k ++){
                bltableLocal(k) = j;
            }
            start = i;
            j ++;
        }

        Duplicate(normalizationLocal, (T)15, TILE_LEN_C);// 00000000 00000000 00000000 00001111
        Duplicate(clearLocal, (T)65535, TILE_LEN_C);// 00000000 00000000 11111111 11111111

        uint64_t compressedSize = 0;// 字节为单位
        uint32_t totalcompressedSize = 0;// 字节为单位
        for(uint32_t i = blockId; i < datablockNum; i += blockNum){
            uint32_t offset = i * TILE_LEN_C;
            CopyIn(offset);
            Compute(compressedSize, tempLocal, encodedLocal, tableLocal, bltableLocal, blLocal, mblLocal, mblbroadcastLocal, powerLocal, multiplierLocal, 
                    mergeLocal, cmblLocal, maskLocal, normalizationLocal, gatherLocal, clearLocal);
            compressedSize = compressedSize * sizeof(uint16_t);// 转换为字节为单位
            CopyOut(totalcompressedSize, compressedSize, i);// 拷贝结果到GM，这里拷贝到字节数必定为32字节的倍数
            totalcompressedSize = totalcompressedSize + (uint32_t)compressedSize;// 更新totalcompressedSize
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t offset) {
        LocalTensor<T> inLocal = inQueue.AllocTensor<T>();
        // 拷贝当前Tile数据到Local
        DataCopy(inLocal, e_input[offset], TILE_LEN_C);
        inQueue.EnQue(inLocal);
    }

    __aicore__ inline void Compute(uint64_t& compressedSize,         
                                   LocalTensor<T>& tempLocal,// TILE_LEN_C
                                   LocalTensor<T>& encodedLocal,
                                   LocalTensor<T>& tableLocal,// HISTOGRAM_BINS
                                   LocalTensor<T>& bltableLocal,// HISTOGRAM_BINS
                                   LocalTensor<T>& blLocal,// TILE_LEN_C
                                   LocalTensor<T>& mblLocal,// NUM * 2
                                   LocalTensor<T>& mblbroadcastLocal,// TILE_LEN_C
                                   LocalTensor<T>& powerLocal,// HISTOGRAM_BINS
                                   LocalTensor<T>& multiplierLocal,// TILE_LEN_C
                                   LocalTensor<T>& mergeLocal,// TILE_LEN_C
                                   LocalTensor<T>& cmblLocal,// TILE_LEN_C
                                   LocalTensor<T>& maskLocal,// TILE_LEN_C / sizeof(uint8_t)
                                   LocalTensor<T>& normalizationLocal,// TILE_LEN_C
                                   LocalTensor<T>& gatherLocal,// TILE_LEN_C
                                   LocalTensor<T>& clearLocal// TILE_LEN_C
    ) {

        LocalTensor<T> e_inLocal = inQueue.DeQue<T>();// TILE_LEN_C
        LocalTensor<T> e_outLocal = e_outQueue.AllocTensor<T>();// TILE_LEN_C
        LocalTensor<T> mbl_outLocal = mbl_outQueue.AllocTensor<T>();// NUM * 2 / 8

        // gather得到编码后的数据
        ShiftLeft(tempLocal, e_inLocal, (uint32_t)2, TILE_LEN_C);//左移2位相当于乘sizeof(uint32_t)，因为gather偏置是字节为单位
        Gather(encodedLocal, tableLocal, tempLocal, (uint32_t)0, (uint32_t)TILE_LEN_C);

        // gather获得编码后数据的bl
        ShiftLeft(tempLocal, encodedLocal, (uint32_t)2, TILE_LEN_C);//左移2位相当于乘sizeof(uint32_t)，因为gather偏置是字节为单位，必须与T的位宽对齐，不然出现未知错误
        Gather(blLocal, bltableLocal, tempLocal, (uint32_t)0, (uint32_t)TILE_LEN_C);//gather比特长度

        // 获取每16个元素的mbl
        WholeReduceMax<float>(mblLocal.template ReinterpretCast<float>(), blLocal.template ReinterpretCast<float>(), TILE_LEN, TILE_LEN_C / TILE_LEN, 1, 1, TILE_LEN * 2 / 32, ReduceOrder::ORDER_ONLY_VALUE);
        Broadcast<float, 2, 1>(mblbroadcastLocal.template ReinterpretCast<float>(), mblLocal.template ReinterpretCast<float>(), dstShape_, srcShape_);
        // 保存mbl
        ShiftLeft(tempLocal, mblLocal[NUM], (uint32_t)4, NUM);
        Or(mblLocal, mblLocal, tempLocal, (int32_t)NUM * 2);
        ShiftLeft(tempLocal, mblLocal[NUM/2], (uint32_t)8, NUM / 2);
        Or(mblLocal, mblLocal, tempLocal, (int32_t)NUM);
        ShiftLeft(tempLocal, mblLocal[NUM/4], (uint32_t)16, NUM / 4);
        Or(mbl_outLocal, mblLocal, tempLocal, (int32_t)NUM / 2);
        
        // gather获取mergeLocal对应乘数，通过整数乘法实现左移不同比特数
        ShiftLeft(tempLocal, mblbroadcastLocal, (uint32_t)2, TILE_LEN_C);
        Gather(multiplierLocal, powerLocal, tempLocal, (uint32_t)0, (uint32_t)TILE_LEN_C);

        // 乘法实现不同通道移位不同比特数，同时使用Or进行低位添加最新bits
        Mul(mergeLocal.template ReinterpretCast<int32_t>(), mergeLocal.template ReinterpretCast<int32_t>(), multiplierLocal.template ReinterpretCast<int32_t>(), (int32_t)TILE_LEN_C);
        Or(mergeLocal, mergeLocal, encodedLocal, (int32_t)TILE_LEN_C * 2);

        // 更新cmbl，同时判断哪些cml>=16，给出一个掩码
        Add(cmblLocal.template ReinterpretCast<int32_t>(), cmblLocal.template ReinterpretCast<int32_t>(), mblbroadcastLocal.template ReinterpretCast<int32_t>(), TILE_LEN_C);
        CompareScalar(maskLocal.template ReinterpretCast<uint8_t>(), cmblLocal.template ReinterpretCast<float>(), 
                        (float)16, CMPMODE::GE, TILE_LEN_C);
        And(cmblLocal, cmblLocal, normalizationLocal, TILE_LEN_C * 2);// 00000000 00000000 00000000 00001111:相与就是对16取余
        
        // 通过掩码将对应bits数>=16的通道数据gather，同时获得计数
        GatherMask(gatherLocal.template ReinterpretCast<float>(), mergeLocal.template ReinterpretCast<float>(), 
                        maskLocal.template ReinterpretCast<uint32_t>(), true, TILE_LEN_C, { 1, 1, 1, 0 }, compressedSize);
        // 折半保存提取出的结果, uint32_t只有低16位有效
        And(gatherLocal, gatherLocal, clearLocal, compressedSize * 2);// clearLocal: 00000000 00000000 11111111 11111111
        ShiftLeft(tempLocal, gatherLocal[compressedSize / 2], (uint32_t)16, compressedSize / 2);// 这里comressedSize以uint32_t为单位
        Or(e_outLocal, gatherLocal, tempLocal, compressedSize / 2 * 2);

        // 更新mergeLocal
        ShiftRight(tempLocal, mergeLocal, (uint32_t)16, TILE_LEN_C);
        Select(mergeLocal.template ReinterpretCast<float>(), maskLocal, tempLocal.template ReinterpretCast<float>(), 
            mergeLocal.template ReinterpretCast<float>(), SELMODE::VSEL_TENSOR_TENSOR_MODE, TILE_LEN_C);

        e_outQueue.EnQue(e_outLocal);
        mbl_outQueue.EnQue(mbl_outLocal);
        inQueue.FreeTensor(e_inLocal);
    }

    __aicore__ inline void CopyOut(uint32_t totalcompressedSize, uint32_t compressedSize, uint32_t datablockId) {
        LocalTensor<T> e_outLocal = e_outQueue.DeQue<T>();
        LocalTensor<T> mbl_outLocal = mbl_outQueue.DeQue<T>();
        // 拷贝当前压缩输出数据到GM 
        DataCopy(output[totalcompressedSize / sizeof(T)], e_outLocal, compressedSize / sizeof(T));
        // 拷贝当前mbl数据到GM
        DataCopy(mbl_output[datablockId * NUM * 2 / 8], mbl_outLocal, NUM * 2 / 8);
        e_outQueue.FreeTensor(e_outLocal);
        mbl_outQueue.FreeTensor(mbl_outLocal);
    }

private:
    TPipe* pipe;

    TQue<QuePosition::VECIN, 1> inQueue;
    TQue<QuePosition::VECOUT, 1> e_outQueue;
    TQue<QuePosition::VECOUT, 1> mbl_outQueue;

    TBuf<TPosition::VECCALC> temp;
    TBuf<TPosition::VECCALC> encoded;
    TBuf<TPosition::VECCALC> table;
    TBuf<TPosition::VECCALC> bltable;
    TBuf<TPosition::VECCALC> bl;
    TBuf<TPosition::VECCALC> mbl;
    TBuf<TPosition::VECCALC> mblbroadcast;
    TBuf<TPosition::VECCALC> power;
    TBuf<TPosition::VECCALC> multiplier;
    TBuf<TPosition::VECCALC> merge;
    TBuf<TPosition::VECCALC> cmbl;
    TBuf<TPosition::VECCALC> mask;
    TBuf<TPosition::VECCALC> normalization;
    TBuf<TPosition::VECCALC> gather;
    TBuf<TPosition::VECCALC> clear;

    GlobalTensor<T> e_input;
    GlobalTensor<T> table_input;
    GlobalTensor<T> mbl_output;
    GlobalTensor<T> output;

    uint32_t blockId;
    uint32_t blockNum;
    uint32_t totalUncompressedBytes;
    uint32_t datablockNum;

    uint32_t srcShape_[2];
    uint32_t dstShape_[2];
};
/*
template<typename T>// T =int32_t
class PrefixKernel {
public:
    __aicore__ inline PrefixKernel() {}// 计算独占前缀和

    __aicore__ inline void Init(TPipe* pipe,
                                uint32_t datablockNum,
                                __gm__ uint8_t* tilePrefix,
                                __gm__ uint8_t* compressedSize,// 输入
                                __gm__ uint8_t* compressedSizePrefix// 输出
                                
                                
    ) {
        this->pipe = pipe;
        this->DATA_BLOCK_NUM = datablockNum;
        tileprefix.SetGlobalBuffer((__gm__ T*)(tilePrefix));
        compSize.SetGlobalBuffer((__gm__ T*)(compressedSize));
        output.SetGlobalBuffer((__gm__ T*)(compressedSizePrefix));

        pipe->InitBuffer(inQueue, BUFFER_NUM, TILE_NUM * sizeof(T));
        pipe->InitBuffer(outQueue, BUFFER_NUM, ((DATA_BLOCK_NUM + 31) / 32) * 32 * sizeof(T));
    }

    __aicore__ inline void Process() {
        pipe->InitBuffer(prefixTemp, ((DATA_BLOCK_NUM + 31) / 32) * 32 * sizeof(T));
        LocalTensor<T> prefixLocal = prefixTemp.Get<T>();

        for(int i = 0; i < DATA_BLOCK_NUM; i ++){
            int offset0 = i * TILE_NUM;
            CopyIn(offset0);
            Compute(i, prefixLocal);
        }
        // assert();
        ComputePrefix(prefixLocal);
        CopyOut(prefixLocal);
    }

private:
    __aicore__ inline void CopyIn(int32_t offset) {
        LocalTensor<T> inLocal = inQueue.AllocTensor<T>();
        DataCopy(inLocal, tileprefix[offset], TILE_NUM);
        inQueue.EnQue(inLocal);
    }

    __aicore__ inline void Compute(uint32_t i, LocalTensor<int32_t> prefixLocal) {
        LocalTensor<T> inLocal = inQueue.DeQue<T>();
        prefixLocal(i) = 
        // inLocal(TILE_NUM - 1);
        ((inLocal(TILE_NUM - 1) + 32 - 1) / 32) * 32;
        // if(i == 1)
        // {
        //     assert(prefixLocal(0) == 2048);
        // }
        inQueue.FreeTensor(inLocal);
    }

    __aicore__ inline void ComputePrefix(LocalTensor<int32_t> prefixLocal){
        LocalTensor<T> outLocal = outQueue.AllocTensor<T>();
        outLocal(0) = 0;
        for(int l = 1; l < DATA_BLOCK_NUM; l ++){
            outLocal(l) = outLocal(l - 1) + prefixLocal(l - 1);
        }
        // assert(outLocal(0) == 0);
        // assert(DATA_BLOCK_NUM == 32);
        outQueue.EnQue(outLocal);
    }

    __aicore__ inline void CopyOut(LocalTensor<int32_t> prefixLocal) {
        LocalTensor<T> outLocal = outQueue.DeQue<T>();
        DataCopy(output, outLocal, ((DATA_BLOCK_NUM + 31) / 32) * 32);//向上取到32的倍数
        DataCopy(compSize, prefixLocal, ((DATA_BLOCK_NUM + 31) / 32) * 32);
        outQueue.FreeTensor(outLocal);
    }

private:
    TPipe* pipe;
    TQue<QuePosition::VECIN, 1> inQueue;
    TQue<QuePosition::VECOUT, 1> outQueue;
    TBuf<TPosition::VECCALC> prefixTemp;

    GlobalTensor<T> tileprefix;
    GlobalTensor<T> compSize;
    GlobalTensor<T> output;

    uint32_t DATA_BLOCK_NUM;
};

template<typename T>
class CoalesceKernel {
public:
    __aicore__ inline CoalesceKernel() {} // 生成数据头，紧缩码字，计算压缩率
    // 输入：max_bit_length数组（3bits * block_num），码字（max_bit_length * blockSize）
    // 输出：一整块连续的压缩块，压缩块的大小

    __aicore__ inline void Init(TPipe* pipe,
                                uint32_t dataBlockNum,
                                __gm__ uint8_t* finalCompressedExp, //output
                                __gm__ uint8_t* compressedSize,
                                __gm__ uint8_t* compressedSizePrefix,
                                uint32_t totalUncompressedBytes) {
        this->pipe = pipe;
        this->dataBlockNum = dataBlockNum;
        this->blockId = GetBlockIdx();
        this->blockNum = GetBlockNum();

        input.SetGlobalBuffer((__gm__ T*)(finalCompressedExp + DATA_BLOCK_BYTE_NUM / 2 * dataBlockNum));
        output.SetGlobalBuffer((__gm__ T*)(finalCompressedExp));
        compressedsize.SetGlobalBuffer((__gm__ T*)(compressedSize));
        compressedsizePrefix.SetGlobalBuffer((__gm__ T*)(compressedSizePrefix));

        pipe->InitBuffer(queBind, BUFFER_NUM, DATA_BLOCK_BYTE_NUM / 2);
    }

public:
    __aicore__ inline void Process()
    {
        pipe->InitBuffer(compSize, dataBlockNum * sizeof(T));
        LocalTensor<T> compSizeLocal = compSize.Get<T>();
        DataCopy(compSizeLocal, compressedsize, dataBlockNum);

        pipe->InitBuffer(compPrefix, dataBlockNum * sizeof(T));
        LocalTensor<T> compPrefixLocal = compPrefix.Get<T>();
        DataCopy(compPrefixLocal, compressedsizePrefix, dataBlockNum);

        // pipe->InitBuffer(copy, DATA_BLOCK_BYTE_NUM / 2);
        // LocalTensor<T> bindLocal = copy.Get<T>();
        auto bindLocal = queBind.AllocTensor<T>();
        for(int i = blockId; i < dataBlockNum; i += blockNum){
            uint32_t compSize = compSizeLocal(i);
            uint32_t compSizePrefix = compPrefixLocal(i);//字节为单位
            DataCopy(bindLocal, input[i * DATA_BLOCK_BYTE_NUM / 2 / sizeof(T)], compSize / sizeof(T));
            // CopyIn(i);
            // CopyOut(i);
            DataCopy(output[compSizePrefix / sizeof(T)], bindLocal, compSize / sizeof(T));
            // if(i == 0)
            // assert(compSizePrefix == 0);
            // if(i == 1)
            // assert(compSizePrefix == 1664);
        }
        queBind.FreeTensor(bindLocal);

    }
// private:
//     __aicore__ inline void CopyIn(int i){
//         auto bindLocal = queBind.AllocTensor<T>();
//         DataCopy(bindLocal, input[i * 2048 / sizeof(T)], compressedsize(i) / sizeof(T));
//         queBind.EnQue(bindLocal);
//         // queBind.FreeTensor(bindLocal);
//     }
//     __aicore__ inline void CopyOut(int i){
//         auto bindLocal = queBind.DeQue();
//         DataCopy(output[compressedsizePrefix(i)], bindLocal, compressedsize(i) / sizeof(T));
//         // queBind.FreeTensor(bindLocal);
//     }
private:
    TPipe* pipe;
    TQueBind<TPosition::VECIN, TPosition::VECOUT, BUFFER_NUM> queBind;
    TBuf<TPosition::VECCALC> copy;
    TBuf<TPosition::VECCALC> compSize;
    TBuf<TPosition::VECCALC> compPrefix;

    GlobalTensor<T> input;//输入每个数据块压缩后的GM地址
    GlobalTensor<T> output;//输出每个数据块压缩后的GM地址
    GlobalTensor<T> compressedsize;
    GlobalTensor<T> compressedsizePrefix;

    uint32_t blockId;
    uint32_t blockNum;
    uint32_t dataBlockNum;
};

__global__ __aicore__ void extractbits_and_histogram(
                                uint32_t datablockNum,//数据块数量
                                __gm__ uint8_t* in, 
                                __gm__ uint8_t* tempBuffer, 
                                __gm__ uint8_t* final, 
                                __gm__ uint8_t* histogramDevice, 
                                uint32_t totalUncompressedSize)
{
    TPipe pipe;
    Extractbits_and_histogramKernel<uint32_t> op;
    op.Init(&pipe, datablockNum, in, tempBuffer, final, histogramDevice, totalUncompressedSize);
    op.Process();
}

__global__ __aicore__ void MergeHistogram(__gm__ uint8_t* hist_in,
                                          __gm__ uint8_t* table)
{
    TPipe pipe;
    MergeHistogramKernel<int32_t> op;
    op.Init(&pipe, hist_in, table);
    op.Process();
}
*/

__global__ __aicore__ void comp(uint32_t datablockNum,
                                __gm__ uint8_t* tempBuffer, //e_input
                                __gm__ uint8_t* final, //output
                                __gm__ uint8_t* histogramDevice, //table_input
                                __gm__ uint8_t* compressedSize, // 用于保存每个数据块最后压缩完的块大小
                                uint32_t totalUncompressedBytes )
{
    TPipe pipe;
    CompressKernel<uint32_t> op;
    op.Init(&pipe, datablockNum, tempBuffer, final, histogramDevice, compressedSize, totalUncompressedBytes);
    op.Process();
}
/*
__global__ __aicore__ void calcprefix(uint32_t datablockNum,
                                      __gm__ uint8_t* tilePrefix,
                                      __gm__ uint8_t* compressedSize,// 输入
                                      __gm__ uint8_t* compressedSizePrefix
                                      )
{
    TPipe pipe;
    PrefixKernel<int32_t> op;
    op.Init(&pipe, datablockNum, tilePrefix, compressedSize, compressedSizePrefix);
    op.Process();
}

__global__ __aicore__ void coalesce(uint32_t dataBlockNum,
                                    __gm__ uint8_t* finalCompressedExp, //output
                                    __gm__ uint8_t* compressedSize,
                                    __gm__ uint8_t* compressedSizePrefix,
                                    uint32_t totalUncompressedBytes)
{
    TPipe pipe;
    CoalesceKernel<uint32_t> op;
    op.Init(&pipe, dataBlockNum, finalCompressedExp, compressedSize, compressedSizePrefix, totalUncompressedBytes);
    op.Process();
}
*/

extern "C" double compress(uint32_t datablockNum, void* stream, uint8_t* srcDevice, uint8_t* tempBuffer, uint8_t* final, int32_t* histogramDevice, uint32_t* tilePrefix, uint32_t* compressedSize, uint32_t* compressedSizePrefix, uint32_t totalUncompressedSize) {
    // extractbits_and_histogram<<<BLOCK_NUM, nullptr, stream>>>(datablockNum, srcDevice, tempBuffer, final, histogramDevice, totalUncompressedSize);//提取字节并计算直方图
    // MergeHistogram<<<1, nullptr, stream>>>(reinterpret_cast<uint8_t*>(histogramDevice), final + 32);
    auto start = std::chrono::high_resolution_clock::now();  
    comp<<<BLOCK_NUM, nullptr, stream>>>(datablockNum, tempBuffer, final, reinterpret_cast<uint8_t*>(histogramDevice), reinterpret_cast<uint8_t*>(tilePrefix), totalUncompressedSize);//压缩函数
    // aclrtSynchronizeStream(stream);
    auto end = std::chrono::high_resolution_clock::now();  
    // calcprefix<<<1, nullptr, stream>>>(datablockNum, reinterpret_cast<uint8_t*>(tilePrefix), reinterpret_cast<uint8_t*>(compressedSize), reinterpret_cast<uint8_t*>(compressedSizePrefix));//计算前缀和，用于后续块合并，字节为单位，
    // coalesce<<<BLOCK_NUM, nullptr, stream>>>(datablockNum, final + 32 + HISTOGRAM_BINS * sizeof(uint8_t) + TILE_NUM * datablockNum + DATA_BLOCK_BYTE_NUM / 2 * datablockNum, reinterpret_cast<uint8_t*>(compressedSize), reinterpret_cast<uint8_t*>(compressedSizePrefix), totalUncompressedSize);//纯搬运内核
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1e3;
}


