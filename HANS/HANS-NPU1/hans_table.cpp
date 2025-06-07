#include <chrono>
// #include "acl/acl.h"
#include "kernel_operator.h"

// #include <iostream>
// #include <algorithm>

// #include "hans_utils.h"

using namespace AscendC;

constexpr uint32_t DATA_BLOCK_BYTE_NUM = 4096;// 单位为字节
constexpr int32_t BUFFER_NUM = 2; // 双缓冲
constexpr int32_t BLOCK_NUM = 256;// block的数量，后续再改成AIV的数量
constexpr uint32_t HISTOGRAM_BINS = 256;// 尽可能是2的幂，直方图桶数
constexpr uint32_t TILE_LEN = 16; // 每个Tile处理16个单元(单元指输入数据的类型)
constexpr uint32_t TILE_NUM = DATA_BLOCK_BYTE_NUM / sizeof(uint32_t) / TILE_LEN; // 每个数据块包含TILE_NUM个TILE

constexpr uint32_t NUM = 64;
constexpr uint32_t TILE_LEN_E = TILE_LEN * NUM;// 16 * 64 = 1024, TILE_LEN_E代表在eh内核中一次处理的元素数量
constexpr uint32_t TILE_NUM_E = DATA_BLOCK_BYTE_NUM / sizeof(uint32_t) / TILE_LEN_E;
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

        pipe->InitBuffer(inQueue, BUFFER_NUM, TILE_LEN_E * sizeof(T));
        pipe->InitBuffer(e_outQueue, BUFFER_NUM, TILE_LEN_E * 2 * sizeof(T));
        pipe->InitBuffer(m_s_outQueue, BUFFER_NUM, TILE_LEN_E / 2 * sizeof(T));
        //因为开启了double_buffer，最多只能开四个queue
    }

    __aicore__ inline void Process(uint32_t tempSize) {

        pipe->InitBuffer(calcBuf0, TILE_LEN_E * sizeof(T));
        pipe->InitBuffer(calcBuf1, TILE_LEN_E * sizeof(T));
        pipe->InitBuffer(calcBuf2, TILE_LEN_E * sizeof(T));
        pipe->InitBuffer(tempHist,  HISTOGRAM_BINS * sizeof(int32_t));
        pipe->InitBuffer(histBuffer0, TILE_LEN_E * 2 * sizeof(int16_t));
        pipe->InitBuffer(histBuffer1, TILE_LEN_E * 2 / 8);
        pipe->InitBuffer(histBuffer2, HISTOGRAM_BINS * sizeof(int32_t));
        // pipe->InitBuffer(histBuffer3, HISTOGRAM_BINS * sizeof(int32_t));
        pipe->InitBuffer(mask0, TILE_LEN_E * sizeof(T));
        pipe->InitBuffer(mask1, TILE_LEN_E * sizeof(T));
        pipe->InitBuffer(mask2, TILE_LEN_E * 2 * sizeof(half));
        pipe->InitBuffer(tmpQue, tempSize * sizeof(uint8_t));
        pipe->InitBuffer(sum, TILE_LEN_E * 2 * sizeof(half));
        pipe->InitBuffer(work, TILE_LEN_E * 2 * sizeof(half));

        LocalTensor<T> tempLocal0 = calcBuf0.Get<T>();
        LocalTensor<T> tempLocal1 = calcBuf1.Get<T>();
        LocalTensor<T> tempLocal2 = calcBuf2.Get<T>();
        LocalTensor<int32_t> histogram = tempHist.Get<int32_t>();
        LocalTensor<int16_t> histTensor0 = histBuffer0.Get<int16_t>();
        LocalTensor<uint64_t> histTensor1 = histBuffer1.Get<uint64_t>();
        LocalTensor<int32_t> histTensor2 = histBuffer2.Get<int32_t>();
        // LocalTensor<int32_t> histTensor3 = histBuffer3.Get<int32_t>();
        LocalTensor<T> mask0_tensor = mask0.Get<T>();
        LocalTensor<T> mask1_tensor = mask1.Get<T>();
        LocalTensor<half> mask2_tensor = mask2.Get<half>();
        LocalTensor<half> sumLocal = sum.Get<half>();
        LocalTensor<half> workLocal = work.Get<half>();
        // LocalTensor<uint8_t> sharedTmpBuffer = tmpQue.Get<uint8_t>();

        Duplicate(histogram, (int32_t)0, HISTOGRAM_BINS);// 初始化全0
        Duplicate(mask0_tensor, (T)16711935, TILE_LEN_E);//00000000 11111111 00000000 11111111
        Duplicate(mask1_tensor, (T)255, TILE_LEN_E);//00000000 00000000 00000000 11111111
        Duplicate(mask2_tensor, (half)1, TILE_LEN_E * 2);//00000000 00000000 00000000 000000001
        // Duplicate(histTensor3, (int32_t)TILE_LEN_E * 2, HISTOGRAM_BINS);

        //该版本直接传一个datablock
        for(int i = blockId; i < datablockNum; i += blockNum){
            int offset = i * DATA_BLOCK_BYTE_NUM / sizeof(T);
            CopyIn(offset);
            Compute(mask0_tensor, mask1_tensor, mask2_tensor, histogram, histTensor0, histTensor1, histTensor2, tempLocal0, tempLocal1, tempLocal2, sumLocal, workLocal);
            CopyOut(offset);
        }
        DataCopy(hist_output, histogram, HISTOGRAM_BINS);
    }

private:
    __aicore__ inline void CopyIn(int32_t offset) {
        LocalTensor<T> inLocal = inQueue.AllocTensor<T>();
        DataCopy(inLocal, input[offset], TILE_LEN_E);
        inQueue.EnQue(inLocal);
    }

    __aicore__ inline void Compute( LocalTensor<T> mask0_tensor,
                                    LocalTensor<T> mask1_tensor,
                                    LocalTensor<half> mask2_tensor,
                                    LocalTensor<int32_t>& histogram,
                                    LocalTensor<int16_t>& histTensor0,
                                    LocalTensor<uint64_t>& histTensor1,
                                    LocalTensor<int32_t>& histTensor2,
                                    // LocalTensor<int32_t>& histTensor3,
                                    LocalTensor<T>& tempLocal0,
                                    LocalTensor<T>& tempLocal1,
                                    LocalTensor<T>& tempLocal2,
                                    LocalTensor<half>& sumLocal,
                                    LocalTensor<half>& workLocal
                                    // LocalTensor<uint8_t>& sharedTmpBuffer
                                    ) {
        LocalTensor<T> inLocal = inQueue.DeQue<T>();
        LocalTensor<T> e_outLocal = e_outQueue.AllocTensor<T>();
        LocalTensor<T> m_s_outLocal = m_s_outQueue.AllocTensor<T>();

        ShiftLeft(tempLocal0, inLocal, (uint32_t)1, TILE_LEN_E);
        ShiftRight(tempLocal1, inLocal, (uint32_t)31, TILE_LEN_E);//int类型自动算数移位,uint32_t为逻辑移位
        Or(tempLocal2, tempLocal0, tempLocal1, (int32_t)TILE_LEN_E * 2);//将sign放在最后
        // assert(tempLocal2(1023) == 1791781580);
        // assert(tempLocal2(511) == 2130738944);
        // assert(tempLocal2(512) == 2130738944);

        And(// m_s_outLocal
            tempLocal0
            , tempLocal2, mask0_tensor, (int32_t)TILE_LEN_E * 2);//取出从高到低1和3字节，尾数部分，mask0_tensor:00000000 11111111 00000000 11111111
        // assert(tempLocal0(1023) == 13369548);
        // assert(tempLocal0(511) == 0);
        // assert(tempLocal0(512) == 0);
        // assert(tempLocal0(0) == 13435084);
        ShiftLeft(// m_s_outLocal
            tempLocal1
            , tempLocal0[TILE_LEN_E / 2], (uint32_t)8, (uint32_t)(TILE_LEN_E / 2));
        // assert(tempLocal1(511) == 3422604288);// 1023 << 8
        // assert(tempLocal1(0) == 0);// 512 << 8
        Or(m_s_outLocal, tempLocal0, tempLocal1, (int32_t)(TILE_LEN_E / 2) * 2);// 对半折叠存储,Or只支持每个元素为16位，但是Or通道之间互通，所以32位的Or可以通过将操作的元素数量✖️2实现，同样And也是
        // assert(m_s_outLocal(0) == 13435084);// 0 = 0 | 512 << 8
        // assert(m_s_outLocal(511) == );// 511 = 511 | 1023 << 8

        ShiftRight(tempLocal1, tempLocal2, (uint32_t)8, (uint32_t)TILE_LEN_E);//右移8位
        And(tempLocal0, tempLocal1, mask0_tensor, (int32_t)TILE_LEN_E * 2);//取出从高到低0和2字节，指数部分，mask0_tensor:00000000 11111111 00000000 11111111
        And(e_outLocal, tempLocal0, mask1_tensor, (int32_t)TILE_LEN_E * 2);//mask1_tensor:00000000 00000000 00000000 11111111
        ShiftRight(tempLocal2, tempLocal0, (uint32_t)16, (uint32_t)TILE_LEN_E);//右移16位
        And(e_outLocal[TILE_LEN_E], tempLocal2, mask1_tensor, (int32_t)TILE_LEN_E * 2);
        // assert(e_outLocal(1024)==106);
        // assert(e_outLocal(2047)==106);
// #pragma unroll
        for(int i = 0; i < HISTOGRAM_BINS; i ++){
            Duplicate(histTensor0, (int16_t)i, TILE_LEN_E * 2);//histTensor0必须为int16_t类型
            Compare(histTensor1.template ReinterpretCast<uint8_t>(), tempLocal0.template ReinterpretCast<half>(), 
                    histTensor0.template ReinterpretCast<half>(), CMPMODE::EQ, TILE_LEN_E * 2);
            Select(histTensor0.template ReinterpretCast<half>(), histTensor1, mask2_tensor, static_cast<half>(0), 
                   AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, TILE_LEN_E * 2);
            ReduceSum<half>(sumLocal, histTensor0.template ReinterpretCast<half>(), workLocal, TILE_LEN_E * 2);
            histTensor2(i) = (int32_t)sumLocal(0);
            // for(int j = 0; j < TILE_LEN_E * 2 / 64; j ++){
            //     histTensor2(i) = histTensor2(i) + (int32_t)ScalarGetCountOfValue<1>(histTensor1(j));//histTensor3必须是int32_t类型
            // }
        }
        Add(histogram, histogram, histTensor2, (int32_t)HISTOGRAM_BINS);//histogram和histTensor3都必须是int32_t类型

        inQueue.FreeTensor(inLocal);
        e_outQueue.EnQue(e_outLocal);
        m_s_outQueue.EnQue(m_s_outLocal);
    }

    __aicore__ inline void CopyOut(int32_t offset) {
        LocalTensor<T> e_outLocal = e_outQueue.DeQue<T>();
        LocalTensor<T> m_s_outLocal = m_s_outQueue.DeQue<T>();

        DataCopy(e_output[offset * 2], e_outLocal, TILE_LEN_E * 2);
        DataCopy(m_s_output[offset / 2], m_s_outLocal, TILE_LEN_E / 2);// 对半折叠

        e_outQueue.FreeTensor(e_outLocal);
        m_s_outQueue.FreeTensor(m_s_outLocal);
    }

private:
    TPipe* pipe;
    TQue<QuePosition::VECIN, 1> inQueue;// 1代表队列的深度
    TQue<QuePosition::VECOUT, 1> e_outQueue;
    TQue<QuePosition::VECOUT, 1> m_s_outQueue;

    TBuf<TPosition::VECCALC> calcBuf0;
    TBuf<TPosition::VECCALC> calcBuf1;
    TBuf<TPosition::VECCALC> calcBuf2;
    TBuf<TPosition::VECCALC> tempHist;
    TBuf<TPosition::VECCALC> histBuffer0;
    TBuf<TPosition::VECCALC> histBuffer1;
    TBuf<TPosition::VECCALC> histBuffer2;
    // TBuf<TPosition::VECCALC> histBuffer3;
    TBuf<TPosition::VECCALC> mask0;
    TBuf<TPosition::VECCALC> mask1;
    TBuf<TPosition::VECCALC> mask2;
    TBuf<TPosition::VECCALC> tmpQue;
    TBuf<TPosition::VECCALC> sum;
    TBuf<TPosition::VECCALC> work;

    GlobalTensor<T> input;
    GlobalTensor<T> e_output;
    GlobalTensor<T> m_s_output;
    GlobalTensor<int32_t> hist_output;

    uint32_t blockId;
    uint32_t blockNum;
    uint32_t datablockNum;
};


template<typename T>//T = uint32_t
class Extractbits_and_histogramKernel1 {
public:
    __aicore__ inline Extractbits_and_histogramKernel1() {} // 切分数据，分离指数位，同时进行histogram统计
    // 输入：uint16_t数组(两两组成一个int32_t)
    // 输出：指数数组（int32_t），尾数+sign数组（uint8_t)，histogram统计数组（int32_t）

    __aicore__ inline void Init(TPipe* pipe,
                                uint32_t datablockNum,
                                __gm__ uint8_t* in, 
                                __gm__ uint8_t* tempBuffer, 
                                __gm__ uint8_t* final, 
                                __gm__ uint8_t* histogramDevice, 
                                uint32_t totalUncompressedSize,
                                uint32_t tempsize) {
        this->pipe = pipe;
        
        this->blockId = GetBlockIdx();
        this->blockNum = GetBlockNum();
        this->datablockNum = datablockNum;
        this->tempSize = tempsize;

        input.SetGlobalBuffer((__gm__ T*)(in));
        e_output.SetGlobalBuffer((__gm__ T*)(tempBuffer));
        m_s_output.SetGlobalBuffer((__gm__ T*)(final + 32 + HISTOGRAM_BINS * sizeof(uint8_t) + TILE_NUM * datablockNum));// 32字节对齐
        hist_output.SetGlobalBuffer((__gm__ int32_t*)(histogramDevice + sizeof(int32_t) * HISTOGRAM_BINS * blockId));

        pipe->InitBuffer(inQueue, BUFFER_NUM, TILE_LEN_E * sizeof(T));
        pipe->InitBuffer(e_outQueue, BUFFER_NUM, TILE_LEN_E * 2 * sizeof(T));
        pipe->InitBuffer(m_s_outQueue, BUFFER_NUM, TILE_LEN_E / 2 * sizeof(T));
        //因为开启了double_buffer，最多只能开四个queue
    }

    __aicore__ inline void Process() {

        pipe->InitBuffer(calcBuf0, TILE_LEN_E * sizeof(T));
        pipe->InitBuffer(calcBuf1, TILE_LEN_E * sizeof(T));
        pipe->InitBuffer(calcBuf2, TILE_LEN_E * sizeof(T));
        pipe->InitBuffer(tempHist,  HISTOGRAM_BINS * sizeof(int32_t));
        pipe->InitBuffer(histBuffer0, TILE_LEN_E * 2 * sizeof(int16_t));
        pipe->InitBuffer(histBuffer1, TILE_LEN_E * 2 * sizeof(int16_t));
        pipe->InitBuffer(histBuffer2, HISTOGRAM_BINS * sizeof(int32_t));
        // pipe->InitBuffer(histBuffer3, HISTOGRAM_BINS * sizeof(int32_t));
        pipe->InitBuffer(mask0, TILE_LEN_E * sizeof(T));
        pipe->InitBuffer(mask1, TILE_LEN_E * sizeof(T));
        pipe->InitBuffer(mask2, TILE_LEN_E * 2 * sizeof(half));
        pipe->InitBuffer(tmpQue, tempSize * sizeof(int16_t));
        pipe->InitBuffer(sum, TILE_LEN_E * 2 * sizeof(half));
        pipe->InitBuffer(work, TILE_LEN_E * 2 * sizeof(half));

        LocalTensor<T> tempLocal0 = calcBuf0.Get<T>();
        LocalTensor<T> tempLocal1 = calcBuf1.Get<T>();
        LocalTensor<T> tempLocal2 = calcBuf2.Get<T>();
        LocalTensor<int32_t> histogram = tempHist.Get<int32_t>();
        LocalTensor<int16_t> histTensor0 = histBuffer0.Get<int16_t>();
        LocalTensor<int16_t> histTensor1 = histBuffer1.Get<int16_t>();
        LocalTensor<int32_t> histTensor2 = histBuffer2.Get<int32_t>();
        // LocalTensor<int32_t> histTensor3 = histBuffer3.Get<int32_t>();
        LocalTensor<T> mask0_tensor = mask0.Get<T>();
        LocalTensor<T> mask1_tensor = mask1.Get<T>();
        LocalTensor<half> mask2_tensor = mask2.Get<half>();
        LocalTensor<half> sumLocal = sum.Get<half>();
        LocalTensor<half> workLocal = work.Get<half>();
        LocalTensor<int16_t> sharedTmpBuffer = tmpQue.Get<int16_t>();

        Duplicate(histogram, (int32_t)0, HISTOGRAM_BINS);// 初始化全0
        Duplicate(mask0_tensor, (T)16711935, TILE_LEN_E);//00000000 11111111 00000000 11111111
        Duplicate(mask1_tensor, (T)255, TILE_LEN_E);//00000000 00000000 00000000 11111111
        Duplicate(mask2_tensor, (half)1, TILE_LEN_E * 2);//00000000 00000000 00000000 000000001

        CreateVecIndex(histTensor0, (int16_t)0, HISTOGRAM_BINS);

        //该版本直接传一个datablock
        for(int i = blockId; i < datablockNum; i += blockNum){
            int offset = i * DATA_BLOCK_BYTE_NUM / sizeof(T);
            CopyIn(offset);
            Compute(mask0_tensor, mask1_tensor, mask2_tensor, histogram, histTensor0, histTensor1, histTensor2, tempLocal0, tempLocal1, tempLocal2, sumLocal, workLocal, sharedTmpBuffer);
            CopyOut(offset);
        }
        DataCopy(hist_output, histogram, HISTOGRAM_BINS);
    }

private:
    __aicore__ inline void CopyIn(int32_t offset) {
        LocalTensor<T> inLocal = inQueue.AllocTensor<T>();
        DataCopy(inLocal, input[offset], TILE_LEN_E);
        inQueue.EnQue(inLocal);
    }

    __aicore__ inline void Compute( LocalTensor<T> mask0_tensor,
                                    LocalTensor<T> mask1_tensor,
                                    LocalTensor<half> mask2_tensor,
                                    LocalTensor<int32_t>& histogram,
                                    LocalTensor<int16_t>& histTensor0,
                                    LocalTensor<int16_t>& histTensor1,
                                    LocalTensor<int32_t>& histTensor2,
                                    // LocalTensor<int32_t>& histTensor3,
                                    LocalTensor<T>& tempLocal0,
                                    LocalTensor<T>& tempLocal1,
                                    LocalTensor<T>& tempLocal2,
                                    LocalTensor<half>& sumLocal,
                                    LocalTensor<half>& workLocal,
                                    LocalTensor<int16_t>& tmpLocal
                                    ) {
        LocalTensor<T> inLocal = inQueue.DeQue<T>();
        LocalTensor<T> e_outLocal = e_outQueue.AllocTensor<T>();
        LocalTensor<T> m_s_outLocal = m_s_outQueue.AllocTensor<T>();

        ShiftLeft(tempLocal0, inLocal, (uint32_t)1, TILE_LEN_E);
        ShiftRight(tempLocal1, inLocal, (uint32_t)31, TILE_LEN_E);//int类型自动算数移位,uint32_t为逻辑移位
        Or(tempLocal2, tempLocal0, tempLocal1, (int32_t)TILE_LEN_E * 2);//将sign放在最后

        And(tempLocal0, tempLocal2, mask0_tensor, (int32_t)TILE_LEN_E * 2);//取出从高到低1和3字节，尾数部分，mask0_tensor:00000000 11111111 00000000 11111111
        ShiftLeft(tempLocal1, tempLocal0[TILE_LEN_E / 2], (uint32_t)8, (uint32_t)(TILE_LEN_E / 2));
        Or(m_s_outLocal, tempLocal0, tempLocal1, (int32_t)(TILE_LEN_E / 2) * 2);// 对半折叠存储,Or只支持每个元素为16位，但是Or通道之间互通，所以32位的Or可以通过将操作的元素数量✖️2实现，同样And也是

        ShiftRight(tempLocal1, tempLocal2, (uint32_t)8, (uint32_t)TILE_LEN_E);//右移8位
        And(tempLocal0, tempLocal1, mask0_tensor, (int32_t)TILE_LEN_E * 2);//取出从高到低0和2字节，指数部分，mask0_tensor:00000000 11111111 00000000 11111111
        And(e_outLocal, tempLocal0, mask1_tensor, (int32_t)TILE_LEN_E * 2);//mask1_tensor:00000000 00000000 00000000 11111111
        ShiftRight(tempLocal2, tempLocal0, (uint32_t)16, (uint32_t)TILE_LEN_E);//右移16位
        And(e_outLocal[TILE_LEN_E], tempLocal2, mask1_tensor, (int32_t)TILE_LEN_E * 2);

        uint64_t rsvdCnt = 0;
        uint32_t mask = TILE_LEN_E * 2;
        for(int i = 0; i < HISTOGRAM_BINS; i ++){
            CompareScalar(histTensor1.template ReinterpretCast<uint8_t>(), tempLocal0.template ReinterpretCast<half>(), 
                            (histTensor0.template ReinterpretCast<half>())(i), CMPMODE::EQ, TILE_LEN_E * 2);
            GatherMask(tempLocal1.template ReinterpretCast<half>(), tempLocal0.template ReinterpretCast<half>(), 
                            histTensor1.template ReinterpretCast<uint16_t>(), true, mask, { 1, 1, 1, 0 }, rsvdCnt);
            histTensor2(i) = rsvdCnt;
        }
        Add(histogram, histogram, histTensor2, (int32_t)HISTOGRAM_BINS);//histogram和histTensor3都必须是int32_t类型

        inQueue.FreeTensor(inLocal);
        e_outQueue.EnQue(e_outLocal);
        m_s_outQueue.EnQue(m_s_outLocal);
    }

    __aicore__ inline void CopyOut(int32_t offset) {
        LocalTensor<T> e_outLocal = e_outQueue.DeQue<T>();
        LocalTensor<T> m_s_outLocal = m_s_outQueue.DeQue<T>();

        DataCopy(e_output[offset * 2], e_outLocal, TILE_LEN_E * 2);
        DataCopy(m_s_output[offset / 2], m_s_outLocal, TILE_LEN_E / 2);// 对半折叠

        e_outQueue.FreeTensor(e_outLocal);
        m_s_outQueue.FreeTensor(m_s_outLocal);
    }

private:
    TPipe* pipe;
    TQue<QuePosition::VECIN, 1> inQueue;// 1代表队列的深度
    TQue<QuePosition::VECOUT, 1> e_outQueue;
    TQue<QuePosition::VECOUT, 1> m_s_outQueue;

    TBuf<TPosition::VECCALC> calcBuf0;
    TBuf<TPosition::VECCALC> calcBuf1;
    TBuf<TPosition::VECCALC> calcBuf2;
    TBuf<TPosition::VECCALC> tempHist;
    TBuf<TPosition::VECCALC> histBuffer0;
    TBuf<TPosition::VECCALC> histBuffer1;
    TBuf<TPosition::VECCALC> histBuffer2;
    // TBuf<TPosition::VECCALC> histBuffer3;
    TBuf<TPosition::VECCALC> mask0;
    TBuf<TPosition::VECCALC> mask1;
    TBuf<TPosition::VECCALC> mask2;
    TBuf<TPosition::VECCALC> tmpQue;
    TBuf<TPosition::VECCALC> sum;
    TBuf<TPosition::VECCALC> work;

    GlobalTensor<T> input;
    GlobalTensor<T> e_output;
    GlobalTensor<T> m_s_output;
    GlobalTensor<int32_t> hist_output;

    uint32_t blockId;
    uint32_t blockNum;
    uint32_t datablockNum;
    uint32_t tempSize;
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
        // assert(inLocal(0) == 0);
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
                                uint32_t totalUncompressedSize,
                                uint32_t tempSize)
{
    TPipe pipe;
    Extractbits_and_histogramKernel1<uint32_t> op;
    op.Init(&pipe, datablockNum, in, tempBuffer, final, histogramDevice, totalUncompressedSize, tempSize);
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


extern "C" void table(uint32_t datablockNum, void* stream, uint8_t* srcDevice, uint8_t* tempBuffer, uint8_t* final, int32_t* histogramDevice, uint32_t* tilePrefix, uint32_t* compressedSize, uint32_t* compressedSizePrefix, uint32_t totalUncompressedSize, uint32_t tempSize) {
    // auto start = std::chrono::high_resolution_clock::now();  
    extractbits_and_histogram<<<BLOCK_NUM, nullptr, stream>>>(datablockNum, srcDevice, tempBuffer, final, histogramDevice, totalUncompressedSize, tempSize);//提取字节并计算直方图
    // CHECK_ACL(aclrtSynchronizeStream(stream));
    // auto end = std::chrono::high_resolution_clock::now();  
    MergeHistogram<<<1, nullptr, stream>>>(reinterpret_cast<uint8_t*>(histogramDevice), final + 32);
}
