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

extern "C" void merge(uint32_t datablockNum, void* stream, uint8_t* srcDevice, uint8_t* tempBuffer, uint8_t* final, int32_t* histogramDevice, uint32_t* tilePrefix, uint32_t* compressedSize, uint32_t* compressedSizePrefix, uint32_t totalUncompressedSize) {
    // extractbits_and_histogram<<<BLOCK_NUM, nullptr, stream>>>(datablockNum, srcDevice, tempBuffer, final, histogramDevice, totalUncompressedSize);//提取字节并计算直方图
    // MergeHistogram<<<1, nullptr, stream>>>(reinterpret_cast<uint8_t*>(histogramDevice), final + 32);
    // comp<<<BLOCK_NUM, nullptr, stream>>>(datablockNum, tempBuffer, final, reinterpret_cast<uint8_t*>(histogramDevice), reinterpret_cast<uint8_t*>(tilePrefix), totalUncompressedSize);//压缩函数
    calcprefix<<<1, nullptr, stream>>>(datablockNum, reinterpret_cast<uint8_t*>(tilePrefix), reinterpret_cast<uint8_t*>(compressedSize), reinterpret_cast<uint8_t*>(compressedSizePrefix));//计算前缀和，用于后续块合并，字节为单位，
    coalesce<<<BLOCK_NUM, nullptr, stream>>>(datablockNum, final + 32 + HISTOGRAM_BINS * sizeof(uint8_t) + TILE_NUM * datablockNum + DATA_BLOCK_BYTE_NUM / 2 * datablockNum, reinterpret_cast<uint8_t*>(compressedSize), reinterpret_cast<uint8_t*>(compressedSizePrefix), totalUncompressedSize);//纯搬运内核
}
