// #include "hans_utils.h"
#include "kernel_operator.h"

// using namespace AscendC;

// //注意：所有算子的输入与输出尽可能32字节对齐，Add这些底层接口的输入与输出必须32字节对齐

// template<typename T>
// class calcprefix_coalesceKernel {
// public:
//     __aicore__ inline calcprefix_coalesceKernel() {} // 生成数据头，紧缩码字，计算压缩率
//     // 
//     // 输出：一整块连续的压缩块，压缩块的大小

//     __aicore__ inline void Init(TPipe* pipe,
//                                 uint32_t dataBlockNum,
//                                 __gm__ uint8_t* blockCompSize, //output
//                                 __gm__ uint8_t* compexp,
//                                 uint32_t bufferSize) {
//         this->pipe = pipe;
//         this->dataBlockNum = dataBlockNum;
//         this->blockId = GetBlockIdx();
//         this->blockNum = GetBlockNum();
//         this->bufferSize = bufferSize;

//         finalexp.SetGlobalBuffer((__gm__ T*)(compexp));
//         blockCompSize.SetGlobalBuffer((__gm__ T*)(blockCompSize));

//         pipe->InitBuffer(queBind, BUFFER_NUM, bufferSize);
//     }

// public:
//     __aicore__ inline void Process()
//     {
//         pipe->InitBuffer(compSize, BLOCK_NUM * 32);
//         LocalTensor<T> compSizeLocal = compSize.Get<T>();

//         pipe->InitBuffer(compSizePrefix, Align8_BLOCK_NUM * sizeof(T));
//         LocalTensor<T> compSizePrefixLocal = compSizePrefix.Get<T>();

//         DataCopy(compSizeLocal, blockCompSize, BLOCK_NUM * 32 / sizeof(T));

//         compSizePrefixLocal(0) = 0;
//         for(int i = 1; i < BLOCK_NUM; i ++){
//             compSizeLocal(i) = compSizeLocal[i * 32 / sizeof(T)](0);
//             compSizePrefixLocal(i) = compSizePrefixLocal(i - 1) + compSizeLocal(i - 1);
//         }

//         uint32_t compSize = compSizeLocal(blockId);// 字节为单位
//         auto bindLocal = queBind.AllocTensor<uint8_t>();
//         DataCopy(bindLocal, retrenchInput[blockId * (datablockNumPerBlock * DATA_BLOCK_BYTE_NUM / 2)], compSize);
//         DataCopy(retrenchOutput[compSizePrefixLocal(blockId)], bindLocal, compSize);

//         queBind.FreeTensor(bindLocal);
//     }

// private:
//     TPipe* pipe;
//     TQueBind<TPosition::VECIN, TPosition::VECOUT, BUFFER_NUM> queBind;

//     TBuf<TPosition::VECCALC> compSize;
//     TBuf<TPosition::VECCALC> compSizePrefix;

//     GlobalTensor<T> finalexp;// 最终压缩块的GM地址
//     GlobalTensor<T> blockCompSize;

//     uint32_t blockId;
//     uint32_t blockNum;
//     uint32_t dataBlockNum;
//     uint32_t bufferSize; // 每个线程block处理的datablock数量
// };

// __global__ __aicore__ void calcprefix_coalesce(uint32_t datablockNum,
//                                       __gm__ uint8_t* blockCompSize,
//                                       __gm__ uint8_t* compexp, //output
//                                       uint32_t bufferSize
//                                       )
// {
//     TPipe pipe;
//     calcprefix_coalesceKernel<int32_t> op;
//     op.Init(&pipe, datablockNum, blockCompSize, compexp, bufferSize);
//     op.Process();
// }

// extern "C" void merge(Header *cphd, void *stream, uint8_t* srcDevice, uint8_t *compressedDevice, uint8_t* blockCompSizeDevice, uint32_t bufferSize){
//     calcprefix_coalesce<<<BLOCK_NUM, nullptr, stream>>>(cphd.dataBlockNum, blockCompSizeDevice, getCompressed_exp(cphd, compressedDevice, uint32_t bufferSize));//计算前缀和，用于后续块合并，字节为单位，
// }
