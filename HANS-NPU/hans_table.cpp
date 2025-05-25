#include "kernel_operator.h"

// #include <iostream>
// #include <algorithm>

// #include "hans_utils.h"

using namespace AscendC;

constexpr uint32_t DATA_BLOCK_BYTE_NUM = 4096;// 单位为字节
constexpr int32_t BUFFER_NUM = 2; // 双缓冲
constexpr int32_t BLOCK_NUM = 256;// block的数量
constexpr uint32_t HISTOGRAM_BINS = 256;// 尽可能是2的幂，直方图桶数
constexpr uint32_t TILE_LEN = 16; // 每个Tile处理32个单元(单元指输入数据的类型)
constexpr uint32_t TILE_NUM = DATA_BLOCK_BYTE_NUM / sizeof(uint32_t) / TILE_LEN; // 每个数据块包含TILE_NUM个TILE

//注意：所有算子的输入与输出尽可能32字节对齐，Add这些底层接口的输入与输出必须32字节对齐

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


extern "C" void table(uint32_t datablockNum, void* stream, uint8_t* srcDevice, uint8_t* tempBuffer, uint8_t* final, int32_t* histogramDevice, uint32_t* tilePrefix, uint32_t* compressedSize, uint32_t* compressedSizePrefix, uint32_t totalUncompressedSize) {
    extractbits_and_histogram<<<BLOCK_NUM, nullptr, stream>>>(datablockNum, srcDevice, tempBuffer, final, histogramDevice, totalUncompressedSize);//提取字节并计算直方图
    MergeHistogram<<<1, nullptr, stream>>>(reinterpret_cast<uint8_t*>(histogramDevice), final + 32);
}
