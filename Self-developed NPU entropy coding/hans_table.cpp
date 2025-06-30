// #include "hans_utils.h"
#include "kernel_operator.h"

using namespace AscendC;
//注意：所有算子的输入与输出尽可能32字节对齐，Add这些底层接口的输入与输出必须32字节对齐

uint32_t DATA_BLOCK_BYTE_NUM_H = 16 * 4096;// 单位为字节,代表E_H内核的分块大小
constexpr uint32_t BUFFER_NUM = 1; // 双缓冲
constexpr uint32_t BLOCK_NUM = 256;// block的数量, 必须是8的倍数
constexpr uint32_t HISTOGRAM_BINS = 256;// 尽可能是2的幂，直方图桶数

template<typename T>//T = uint16_t
class HistogramKernelBF16 {
public:
    __aicore__ inline HistogramKernelBF16() {} // 切分数据，分离指数位，同时进行histogram统计
    // 输入：uint16_t数组(两两组成一个int32_t)
    // 输出：指数数组（int32_t），尾数+sign数组（uint8_t)，histogram统计数组（int32_t）

    __aicore__ inline void Init(TPipe* pipe,
                                uint32_t datablockNum_H,
                                uint32_t lastdataBlockNum, // 总元素数量
                                uint32_t elementNum_H, // 每个数据块的元素数量
                                __gm__ uint8_t* in, 
                                __gm__ uint8_t* histogramDevice
                                ) {
        this->pipe = pipe;
        
        this->blockId = GetBlockIdx();
        this->blockNum = GetBlockNum();
        this->datablockNum_H = datablockNum_H;
        this->lastdataBlockNum = lastdataBlockNum;
        this->elementNum_H = elementNum_H;

        input.SetGlobalBuffer((__gm__ T*)(in));
        hist_output.SetGlobalBuffer((__gm__ int32_t*)(histogramDevice + sizeof(int32_t) * HISTOGRAM_BINS * blockId));

        pipe->InitBuffer(inQueue, BUFFER_NUM, elementNum_H * sizeof(T));
    }

    __aicore__ inline void Process() {

        pipe->InitBuffer(tempHist,  HISTOGRAM_BINS * sizeof(int32_t));
        pipe->InitBuffer(histBuffer0, HISTOGRAM_BINS * sizeof(int16_t));
        pipe->InitBuffer(histBuffer2, HISTOGRAM_BINS * sizeof(int32_t));
        pipe->InitBuffer(mask, elementNum_H / 8);
        pipe->InitBuffer(calcBuf, elementNum_H * sizeof(T));

        LocalTensor<int32_t> histogram = tempHist.Get<int32_t>();
        LocalTensor<int16_t> histTensor0 = histBuffer0.Get<int16_t>();
        LocalTensor<int32_t> histTensor2 = histBuffer2.Get<int32_t>();
        LocalTensor<T> maskLocal = mask.Get<T>();
        LocalTensor<T> tempLocal = calcBuf.Get<T>();

        Duplicate(histogram, (int32_t)0, HISTOGRAM_BINS);// 初始化全0
        CreateVecIndex(histTensor0, (int16_t)0, HISTOGRAM_BINS);

        //该版本直接传一个datablock
        for(int i = blockId; i < datablockNum_H; i += blockNum){
            int offset = i * elementNum_H;
            int computeNum = (i == datablockNum_H - 1) ? lastdataBlockNum : elementNum_H;
            CopyIn(offset);
            Compute(tempLocal, maskLocal, histogram, histTensor0, histTensor2, computeNum);
        }
        DataCopy(hist_output, histogram, HISTOGRAM_BINS);
    }

private:
    __aicore__ inline void CopyIn(int32_t offset) {
        LocalTensor<T> inLocal = inQueue.AllocTensor<T>();
        DataCopy(inLocal, input[offset], elementNum_H);
        inQueue.EnQue(inLocal);
    }

    __aicore__ inline void Compute( LocalTensor<T> tempLocal,
                                    LocalTensor<T> maskLocal,
                                    LocalTensor<int32_t>& histogram,
                                    LocalTensor<int16_t>& histTensor0,
                                    LocalTensor<int32_t>& histTensor2,
                                    uint32_t computeNum
                                    ) {
        LocalTensor<T> inLocal = inQueue.DeQue<T>();

        ShiftLeft(inLocal, inLocal, (uint16_t)1, computeNum);
        ShiftRight(inLocal, inLocal, (uint16_t)8, computeNum);

        uint64_t rsvdCnt = 0;
        for(int i = 0; i < HISTOGRAM_BINS; i ++){
            CompareScalar(maskLocal.template ReinterpretCast<uint8_t>(), inLocal.template ReinterpretCast<half>(), 
                            (histTensor0.template ReinterpretCast<half>())(i), CMPMODE::EQ, computeNum);
            GatherMask(tempLocal, tempLocal, 
                            maskLocal, true, computeNum, { 1, 1, 1, 0 }, rsvdCnt);
            histTensor2(i) = (int32_t)rsvdCnt;
        }
        Add(histogram, histogram, histTensor2, (int32_t)HISTOGRAM_BINS);//histogram和histTensor3都必须是int32_t类型
        
        inQueue.FreeTensor(inLocal);
    }

private:
    TPipe* pipe;
    TQue<QuePosition::VECIN, 1> inQueue;// 1代表队列的深度

    TBuf<TPosition::VECCALC> tempHist;
    TBuf<TPosition::VECCALC> histBuffer0;
    TBuf<TPosition::VECCALC> histBuffer2;
    TBuf<TPosition::VECCALC> mask;
    TBuf<TPosition::VECCALC> calcBuf;

    GlobalTensor<T> input;
    GlobalTensor<int32_t> hist_output;

    uint32_t blockId;
    uint32_t blockNum;
    uint32_t datablockNum_H;
    uint32_t lastdataBlockNum; // 总元素数量
    uint32_t elementNum_H; // 每个数据块的元素数量
};

template<typename T>// int32_t
class MergeHistogramKernelBF16 {
public:
    __aicore__ inline MergeHistogramKernelBF16() {} // 合并blockNum个直方图，生成全局直方图和全局编码表

    __aicore__ inline void Init(TPipe* pipe,
                                __gm__ uint8_t* hist_in) {
        this->pipe = pipe;
        this->blockId = GetBlockIdx();
        this->blockNum = GetBlockNum();

        hist.SetGlobalBuffer((__gm__ T*)(hist_in));

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
        // assert(tempLocal(0) == 0);
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
    }

private:
    __aicore__ inline void CopyIn(uint32_t offset) {
        LocalTensor<T> inLocal = inQueue.AllocTensor<T>();
        DataCopy(inLocal, hist[offset * HISTOGRAM_BINS], HISTOGRAM_BINS);
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
        uint32_t bltable[HISTOGRAM_BINS];
        uint32_t j = 0;
        uint32_t start = 0;
        for(int i = 1; i <= HISTOGRAM_BINS; i <<= 1){
            for(int k = start; k < i; k ++){
                bltable[k] = j;
            }
            start = i;
            j ++;
        }
        for (int i = 0; i < HISTOGRAM_BINS; i++) {
            tableLocal(i) = (uint8_t)tempLocal(i);
        }
        for (int i = 0; i < HISTOGRAM_BINS; i++){
            int32_t temp = tableLocal(i);
            tempLocal(i) = (temp << 14) | (1 << (5 + bltable[temp])) | bltable[temp];
        }
    }

private:
    TPipe* pipe;
    TQue<QuePosition::VECIN, 1> inQueue;

    TBuf<QuePosition::VECCALC> temp;
    TBuf<QuePosition::VECCALC> sorttemp;
    TBuf<QuePosition::VECCALC> tabletemp;

    GlobalTensor<T> hist;

    uint32_t blockId;
    uint32_t blockNum;
};

__global__ __aicore__ void HistogramBF16(
                                uint32_t datablockNum_H,//数据块数量
                                uint32_t lastdataBlockNum, // 总元素数量
                                uint32_t elementNum_H, // 每个数据块的元素数量
                                __gm__ uint8_t* in, 
                                __gm__ uint8_t* histogramDevice
                                )
{
    TPipe pipe;
    HistogramKernelBF16<uint16_t> op;
    op.Init(&pipe, datablockNum_H, lastdataBlockNum, elementNum_H, in, histogramDevice);
    op.Process();
}

__global__ __aicore__ void MergeHistogramBF16(__gm__ uint8_t* hist_in)
{
    TPipe pipe;
    MergeHistogramKernelBF16<int32_t> op;
    op.Init(&pipe, hist_in);
    op.Process();
}

extern "C" void table(uint32_t totalUncompressedSize, void* stream, uint8_t* srcDevice, uint8_t* histogramDevice, uint32_t dataType) {
    switch(dataType) {
        case 0:{ // BF16
            uint32_t datablockNum_H = (totalUncompressedSize + DATA_BLOCK_BYTE_NUM_H - 1) / DATA_BLOCK_BYTE_NUM_H; // 计算数据块数量
            uint32_t totalElementNum0 = totalUncompressedSize / sizeof(uint16_t);
            uint32_t lastdataBlockNum = totalElementNum0 % (DATA_BLOCK_BYTE_NUM_H / sizeof(uint16_t));
            uint32_t elementNum_H0 = DATA_BLOCK_BYTE_NUM_H / sizeof(uint16_t);
            HistogramBF16<<<BLOCK_NUM, nullptr, stream>>>(datablockNum_H, lastdataBlockNum, elementNum_H0, srcDevice, histogramDevice);
            MergeHistogramBF16<<<1, nullptr, stream>>>(histogramDevice);
            break;
        }
        case 1:{ // FP16
            uint32_t totalElementNum1 = totalUncompressedSize / sizeof(uint16_t);
            uint32_t elementNum_H1 = DATA_BLOCK_BYTE_NUM_H / sizeof(uint16_t);
            break;
        }
        case 2:{ // FP32
            uint32_t totalElementNum2 = totalUncompressedSize / sizeof(uint32_t);
            uint32_t elementNum_H2 = DATA_BLOCK_BYTE_NUM_H / sizeof(uint32_t);
            break;
        }
        default:{
            // throw std::runtime_error("Unsupported data type for histogram.");
            break;
        }
    }
}
