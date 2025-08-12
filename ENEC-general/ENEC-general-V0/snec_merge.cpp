/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
 * This file constains code of cpu debug and npu code.We read data from bin file
 * and write result to file.
 */

#include "snec_utils.h"
#include "snec_device.h"

// template<typename T>
// class calcprefix_coalesceKernel {
// public:
//     __aicore__ inline calcprefix_coalesceKernel() {}

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

//         uint32_t compSize = compSizeLocal(blockId);
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

//     GlobalTensor<T> finalexp;
//     GlobalTensor<T> blockCompSize;

//     uint32_t blockId;
//     uint32_t blockNum;
//     uint32_t dataBlockNum;
//     uint32_t bufferSize;
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
//     calcprefix_coalesce<<<BLOCK_NUM, nullptr, stream>>>(cphd.dataBlockNum, blockCompSizeDevice, getCompressed_exp(cphd, compressedDevice, uint32_t bufferSize));
// }
