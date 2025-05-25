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

// template<typename T>//T = uint32_t
// class Extractbits_and_histogramKernel {
// public:
//     __aicore__ inline Extractbits_and_histogramKernel() {} // 切分数据，分离指数位，同时进行histogram统计
//     // 输入：uint16_t数组(两两组成一个int32_t)
//     // 输出：指数数组（int32_t），尾数+sign数组（uint8_t)，histogram统计数组（int32_t）

//     __aicore__ inline void Init(TPipe* pipe,
//                                 uint32_t datablockNum,
//                                 __gm__ uint8_t* in, 
//                                 __gm__ uint8_t* tempBuffer, 
//                                 __gm__ uint8_t* final, 
//                                 __gm__ uint8_t* histogramDevice, 
//                                 uint32_t totalUncompressedSize) {
//         this->pipe = pipe;
        
//         this->blockId = GetBlockIdx();
//         this->blockNum = GetBlockNum();
//         this->datablockNum = datablockNum;

//         input.SetGlobalBuffer((__gm__ T*)(in));
//         e_output.SetGlobalBuffer((__gm__ T*)(tempBuffer));
//         m_s_output.SetGlobalBuffer((__gm__ T*)(final + 32 + HISTOGRAM_BINS * sizeof(uint8_t) + TILE_NUM * datablockNum));// 32字节对齐
//         hist_output.SetGlobalBuffer((__gm__ int32_t*)(histogramDevice + sizeof(int32_t) * HISTOGRAM_BINS * blockId));

//         pipe->InitBuffer(inQueue, BUFFER_NUM, TILE_LEN * sizeof(T));
//         pipe->InitBuffer(e_outQueue, BUFFER_NUM, TILE_LEN * sizeof(T));
//         pipe->InitBuffer(m_s_outQueue, BUFFER_NUM, TILE_LEN / 2 * sizeof(T));
//         //因为开启了double_buffer，最多只能开四个queue
//     }

//     __aicore__ inline void Process() {

//         pipe->InitBuffer(calcBuf0, TILE_LEN * sizeof(T));
//         pipe->InitBuffer(calcBuf1, TILE_LEN * sizeof(T));
//         pipe->InitBuffer(calcBuf2, TILE_LEN * sizeof(T));
//         pipe->InitBuffer(calcBuf3, TILE_LEN * sizeof(T));
//         pipe->InitBuffer(tempHist, TILE_LEN * HISTOGRAM_BINS * sizeof(int32_t));
//         pipe->InitBuffer(histBuffer0, TILE_LEN * sizeof(int32_t));
//         pipe->InitBuffer(histBuffer1, TILE_LEN * sizeof(int32_t));
//         pipe->InitBuffer(mask0, TILE_LEN * sizeof(T));
//         pipe->InitBuffer(mask1, TILE_LEN * sizeof(T));
//         pipe->InitBuffer(mask2, TILE_LEN * sizeof(T));
//         pipe->InitBuffer(mask3, TILE_LEN * sizeof(T));
//         pipe->InitBuffer(mask4, TILE_LEN * sizeof(T));
//         pipe->InitBuffer(offsetBuffer, TILE_LEN * sizeof(int32_t));
//         pipe->InitBuffer(one, TILE_LEN * sizeof(int32_t));

//         LocalTensor<T> tempLocal0 = calcBuf0.Get<T>();
//         LocalTensor<T> tempLocal1 = calcBuf1.Get<T>();
//         LocalTensor<T> tempLocal2 = calcBuf2.Get<T>();
//         LocalTensor<T> tempLocal3 = calcBuf3.Get<T>();
//         LocalTensor<int32_t> histogram = tempHist.Get<int32_t>();
//         LocalTensor<int32_t> histTensor0 = histBuffer0.Get<int32_t>();
//         LocalTensor<int32_t> histTensor1 = histBuffer1.Get<int32_t>();
//         LocalTensor<T> mask0_tensor = mask0.Get<T>();
//         LocalTensor<T> mask1_tensor = mask1.Get<T>();
//         LocalTensor<T> mask2_tensor = mask2.Get<T>();
//         LocalTensor<T> mask3_tensor = mask3.Get<T>();
//         LocalTensor<T> mask4_tensor = mask4.Get<T>();
//         LocalTensor<int32_t> all_one = one.Get<int32_t>();
//         LocalTensor<int32_t> offset_tensor = offsetBuffer.Get<int32_t>();

//         Duplicate(histogram, (int32_t)0, HISTOGRAM_BINS * TILE_LEN);// 初始化全0
//         Duplicate(mask0_tensor, (T)65280, TILE_LEN);//11111111 00000000
//         Duplicate(mask1_tensor, (T)255, TILE_LEN);//00000000 11111111
//         Duplicate(mask2_tensor, (T)16711935, TILE_LEN);//00000000 11111111 00000000 11111111
//         Duplicate(mask3_tensor, (T)4278255360, TILE_LEN);//11111111 00000000 11111111 00000000
//         Duplicate(mask4_tensor, (T)65535, TILE_LEN);//00000000 00000000 11111111 11111111
//         Duplicate(all_one, (int32_t)1, TILE_LEN);//0000 0000 0000 0001
//         uint32_t num = ((1 << 16) + 1) * HISTOGRAM_BINS * sizeof(T);
//         for(int i = 0; i < TILE_LEN; i ++){
//             offset_tensor(i) = i * num;
//         }
//         // printf("datablockNum: %d\n", datablockNum);
//         for(int i = blockId; i < datablockNum; i += blockNum){
//             // if(blockId == 1)
//             // {
//             //     printf("i:%d\n", i);
//             // }
//             int offset0 = i * DATA_BLOCK_BYTE_NUM / sizeof(T);// 指数原大小保存
//             for(int tileIdx = 0; tileIdx < TILE_NUM; tileIdx ++){
//                 int offset1 = tileIdx * TILE_LEN;
//                 int offset = offset0 + offset1;
//                 CopyIn(offset);
//                 Compute(mask0_tensor, mask1_tensor, mask2_tensor, mask3_tensor, mask4_tensor, all_one, offset_tensor, histogram, tempLocal0, tempLocal1, tempLocal2, tempLocal3, histTensor0, histTensor1);
//                 CopyOut(offset);
//             }
//         }
//         MergeLocalHist(histogram);// 合并TILE_LEN个temp直方图为最终的一个
//     }

// private:
//     __aicore__ inline void CopyIn(int32_t offset) {
//         LocalTensor<T> inLocal = inQueue.AllocTensor<T>();

//         DataCopy(inLocal, input[offset], TILE_LEN);
//         inQueue.EnQue(inLocal);
//     }

//     __aicore__ inline void Compute(LocalTensor<T>& mask0_tensor,
//                                    LocalTensor<T>& mask1_tensor,
//                                    LocalTensor<T>& mask2_tensor,
//                                    LocalTensor<T>& mask3_tensor,
//                                    LocalTensor<T>& mask4_tensor,
//                                    LocalTensor<int32_t>& all_one,
//                                    LocalTensor<int32_t>& offset_tensor,
//                                    LocalTensor<int32_t>& histogram,
//                                    LocalTensor<T>& tempLocal0,
//                                    LocalTensor<T>& tempLocal1,
//                                    LocalTensor<T>& tempLocal2,
//                                    LocalTensor<T>& tempLocal3,
//                                    LocalTensor<int32_t>& histTensor0,
//                                    LocalTensor<int32_t>& histTensor1
//                                    ) {
//         LocalTensor<T> inLocal = inQueue.DeQue<T>();
//         LocalTensor<T> e_outLocal = e_outQueue.AllocTensor<T>();
//         LocalTensor<T> m_s_outLocal = m_s_outQueue.AllocTensor<T>();

//         // len /= 2;
//         // 处理每个元素，每次提取32个int32_t（取出64个uint16_t）
//         ShiftLeft(tempLocal0, inLocal, (uint32_t)1, TILE_LEN);
//         ShiftRight(tempLocal1, inLocal, (uint32_t)31, TILE_LEN);//int类型自动算数移位,uint32_t为逻辑移位
//         Or(tempLocal2, tempLocal0, tempLocal1, (int32_t)TILE_LEN * 2);//将sign放在最后

//         And(tempLocal0, tempLocal2, mask2_tensor, (uint32_t)TILE_LEN * 2);//取出从高到低1和3字节，尾数部分
//         ShiftLeft(tempLocal1, tempLocal0[TILE_LEN / 2], (uint32_t)8, (uint32_t)(TILE_LEN / 2));
//         Or(m_s_outLocal, tempLocal0, tempLocal1, (int32_t)TILE_LEN);// 对半折叠存储

//         And(tempLocal3, tempLocal2, mask3_tensor, (int32_t)TILE_LEN * 2);//取出从高到低0和2字节，指数部分
//         ShiftRight(e_outLocal, tempLocal3, (uint32_t)8, (uint32_t)TILE_LEN);//右移8位
//         ShiftRight(tempLocal1, tempLocal3, (uint32_t)(8 - 2), (int32_t)TILE_LEN);// 因为是uint32_t，需要乘四字节，所以少右移2位
//         Add(tempLocal2.template ReinterpretCast<int32_t>(), tempLocal1.template ReinterpretCast<int32_t>(), offset_tensor, (int32_t)TILE_LEN);

//         And(tempLocal0, tempLocal2, mask4_tensor, (int32_t)TILE_LEN * 2);//取出低16位
//         Gather(histTensor0, histogram, tempLocal0, (uint32_t)0, (uint32_t)TILE_LEN);// offset为字节单位
//         Add(histTensor1, histTensor0, all_one, (int32_t)TILE_LEN);
//         for(int i = 0; i < TILE_LEN; i ++){
//             histogram(tempLocal0(i) / sizeof(T)) = histTensor1(i);//需要除sizeof(T)转成T为单位
//         }
//         // Scatter(histogram.template ReinterpretCast<uint32_t>(), histTensor1.template ReinterpretCast<uint32_t>(), tempLocal0, (uint32_t)0, (uint32_t)len);

//         ShiftRight(tempLocal0, tempLocal2, (uint32_t)16, (int32_t)TILE_LEN);//取出高16位
//         Gather(histTensor0, histogram, tempLocal0, (uint32_t)0, (uint32_t)TILE_LEN);
//         Add(histTensor1, histTensor0, all_one, (int32_t)TILE_LEN);
//         for(int i = 0; i < TILE_LEN; i ++){
//             histogram(tempLocal0(i) / sizeof(T)) = histTensor1(i);
//         }
//         // Scatter(histogram, histTensor1, tempLocal0, (uint32_t)0, (uint32_t)len);

//         inQueue.FreeTensor(inLocal);
//         e_outQueue.EnQue(e_outLocal);
//         m_s_outQueue.EnQue(m_s_outLocal);
//     }

//     __aicore__ inline void CopyOut(int32_t offset) {
//         LocalTensor<T> e_outLocal = e_outQueue.DeQue<T>();
//         LocalTensor<T> m_s_outLocal = m_s_outQueue.DeQue<T>();

//         DataCopy(e_output[offset], e_outLocal, TILE_LEN);
//         DataCopy(m_s_output[offset / 2], m_s_outLocal, TILE_LEN / 2);// 对半折叠

//         e_outQueue.FreeTensor(e_outLocal);
//         m_s_outQueue.FreeTensor(m_s_outLocal);
//     }

//     __aicore__ inline void MergeLocalHist(LocalTensor<int32_t>& histogram) {
//         // for(int i = 1; i < TILE_NUM; i ++){
//         //     Add(histogram, histogram, histogram[i * HISTOGRAM_BINS], (int32_t)HISTOGRAM_BINS);
//         // }
//         for(int i = 1; i < TILE_LEN; i ++){
//             for(int j = 0; j < HISTOGRAM_BINS; j ++){
//                 histogram(j) = histogram(j) + histogram(i * HISTOGRAM_BINS + j);
//             }
//         }
//         int sum = 0;
//         for(int i = 0; i < HISTOGRAM_BINS; i ++)
//             sum = sum + histogram(i);
//         // if(blockId == 0) assert(sum == 2048);
//         DataCopy(hist_output, histogram, HISTOGRAM_BINS);
//     }

// private:
//     TPipe* pipe;
//     TQue<QuePosition::VECIN, 1> inQueue;// 1代表队列的深度
//     TQue<QuePosition::VECOUT, 1> e_outQueue;
//     TQue<QuePosition::VECOUT, 1> m_s_outQueue;

//     TBuf<TPosition::VECCALC> calcBuf0;
//     TBuf<TPosition::VECCALC> calcBuf1;
//     TBuf<TPosition::VECCALC> calcBuf2;
//     TBuf<TPosition::VECCALC> calcBuf3;
//     TBuf<TPosition::VECCALC> tempHist;
//     TBuf<TPosition::VECCALC> histBuffer0;
//     TBuf<TPosition::VECCALC> histBuffer1;
//     TBuf<TPosition::VECCALC> mask0;
//     TBuf<TPosition::VECCALC> mask1;
//     TBuf<TPosition::VECCALC> mask2;
//     TBuf<TPosition::VECCALC> mask3;
//     TBuf<TPosition::VECCALC> mask4;
//     TBuf<TPosition::VECCALC> one;
//     TBuf<TPosition::VECCALC> offsetBuffer;

//     GlobalTensor<T> input;
//     GlobalTensor<T> e_output;
//     GlobalTensor<T> m_s_output;
//     GlobalTensor<int32_t> hist_output;

//     uint32_t blockId;
//     uint32_t blockNum;
//     uint32_t datablockNum;
// };

// template<typename T>// int32_t
// class MergeHistogramKernel {
// public:
//     __aicore__ inline MergeHistogramKernel() {} // 合并blockNum个直方图，生成全局直方图和全局编码表

//     __aicore__ inline void Init(TPipe* pipe,
//                                 __gm__ uint8_t* hist_in,
//                                 __gm__ uint8_t* final_table) {
//         this->pipe = pipe;
//         this->blockId = GetBlockIdx();
//         this->blockNum = GetBlockNum();

//         hist.SetGlobalBuffer((__gm__ T*)(hist_in));
//         table.SetGlobalBuffer((__gm__ uint8_t*)(final_table));

//         pipe->InitBuffer(inQueue, BUFFER_NUM, HISTOGRAM_BINS * sizeof(T));
//     }

//     __aicore__ inline void Process() {
//         pipe->InitBuffer(temp, HISTOGRAM_BINS * sizeof(T));
//         LocalTensor<T> tempLocal = temp.Get<T>();
//         Duplicate(tempLocal, (int32_t)0, HISTOGRAM_BINS);

//         pipe->InitBuffer(sorttemp, HISTOGRAM_BINS * sizeof(uint64_t));
//         LocalTensor<uint64_t> sortLocal = sorttemp.Get<uint64_t>();

//         pipe->InitBuffer(tabletemp, HISTOGRAM_BINS * sizeof(uint8_t));
//         LocalTensor<uint8_t> tableLocal = tabletemp.Get<uint8_t>();
//         assert(tempLocal(0) == 0);
//         for(int i = 0; i < //2
//         BLOCK_NUM
//         ; i ++){
//             CopyIn(i);
//             Compute(tempLocal);
//         }

//         Sort(tempLocal, sortLocal);
//         Generate_table(tempLocal, sortLocal, tableLocal);
//         // assert(tempLocal(0) == 0);
//         DataCopy(hist, tempLocal, HISTOGRAM_BINS);
//         DataCopy(table, tableLocal, HISTOGRAM_BINS);
//     }

// private:
//     __aicore__ inline void CopyIn(uint32_t offset) {
//         LocalTensor<T> inLocal = inQueue.AllocTensor<T>();
//         DataCopy(inLocal, hist[offset * HISTOGRAM_BINS], HISTOGRAM_BINS);
//         // if(offset == 1){
//             // assert(inLocal(117) == 1413);
//             // assert(inLocal(119) == 0);
//         // }
//         assert(inLocal(0) == 0);
//         inQueue.EnQue(inLocal);
//     }

//     __aicore__ inline void Compute(LocalTensor<T>& tempLocal){
//         LocalTensor<T> inLocal = inQueue.DeQue<T>();
//         Add(tempLocal, inLocal, tempLocal, (T)HISTOGRAM_BINS);
//         inQueue.FreeTensor(inLocal);
//     }

//     __aicore__ inline void Sort(LocalTensor<T>& tempLocal, LocalTensor<uint64_t> sortLocal){
//         for(int i = 0; i < HISTOGRAM_BINS; i ++){
//             sortLocal(i) = (((uint64_t)tempLocal(i)) << 32) | i;
//         }
//         for (int i = 0; i < HISTOGRAM_BINS - 1; i++) {
//             for (int j = 0; j < HISTOGRAM_BINS - i - 1; j++) {
//                 if (sortLocal(j) < sortLocal(j + 1)) {
//                     uint64_t temp = sortLocal(j);
//                     sortLocal(j) = sortLocal(j + 1);
//                     sortLocal(j + 1) = temp;
//                 }
//             }
//         }
//     }
//     //  如果两个数相同，序号大的在前
//     __aicore__ inline void Generate_table(LocalTensor<T>& tempLocal, LocalTensor<uint64_t>& sortLocal, LocalTensor<uint8_t>& tableLocal){
//         for(int i = 0; i < HISTOGRAM_BINS; i ++){
//             tempLocal(sortLocal(i) & 0xffffffff) = i;
//         }
//         for (int i = 0; i < HISTOGRAM_BINS; i++) {
//             tableLocal(i) = (uint8_t)tempLocal(i);
//         }
//     }

// private:
//     TPipe* pipe;
//     TQue<QuePosition::VECIN, 1> inQueue;
//     TBuf<QuePosition::VECCALC> temp;
//     TBuf<QuePosition::VECCALC> sorttemp;
//     TBuf<QuePosition::VECCALC> tabletemp;

//     GlobalTensor<T> hist;
//     GlobalTensor<uint8_t> table;

//     uint32_t blockId;
//     uint32_t blockNum;
// };

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

        e_input.SetGlobalBuffer((__gm__ uint32_t*)(tempBuffer));
        // assert(e_input(0) == 4653194);
        table_input.SetGlobalBuffer((__gm__ uint32_t*)(histogramDevice));
        mbl_output.SetGlobalBuffer((__gm__ uint8_t*)(final + 32 + HISTOGRAM_BINS));
        output.SetGlobalBuffer((__gm__ uint16_t*)(final + 32 + HISTOGRAM_BINS + TILE_NUM * datablockNum + DATA_BLOCK_BYTE_NUM / 2 * datablockNum + DATA_BLOCK_BYTE_NUM / 2 * datablockNum));
        // compSize.SetGlobalBuffer((__gm__ uint32_t*)(compressedSize));
        tileprefix.SetGlobalBuffer((__gm__ uint32_t*)(tilePrefix));

        pipe->InitBuffer(inQueue, BUFFER_NUM, TILE_LEN * sizeof(T));// 初始化Pipe缓冲区，每个Tile大小TILE_LEN
        // pipe->InitBuffer(e_outQueue, BUFFER_NUM, DATA_BLOCK_BYTE_NUM / 2 * sizeof(uint8_t));
    }

    __aicore__ inline void Process(//uint32_t& compressedSize
                                    ) {
        pipe->InitBuffer(table, HISTOGRAM_BINS * sizeof(uint32_t));
        LocalTensor<uint32_t> tableLocal = table.Get<uint32_t>();
        DataCopy(tableLocal, table_input, HISTOGRAM_BINS);

        pipe->InitBuffer(bits_length, HISTOGRAM_BINS * sizeof(uint32_t));
        LocalTensor<uint32_t> blLocal = bits_length.Get<uint32_t>();
        uint32_t j = 0;
        uint32_t start = 0;
        for(int i = 1; i <= HISTOGRAM_BINS; i <<= 1){
            for(int k = start; k < i; k ++){
                blLocal(k) = j;
            }
            start = i;
            j ++;
        }
        // assert(blLocal(255) == 8);

        pipe->InitBuffer(mask, TILE_LEN * sizeof(uint32_t));
        LocalTensor<uint32_t> maskLocal = mask.Get<uint32_t>();
        Duplicate(maskLocal, (uint32_t)255, TILE_LEN);
        // 0x11111111

        pipe->InitBuffer(calcBuf0, TILE_LEN * sizeof(uint32_t));
        pipe->InitBuffer(calcBuf1, TILE_LEN * sizeof(uint32_t));
        pipe->InitBuffer(calcBuf2, TILE_LEN * sizeof(uint32_t));
        pipe->InitBuffer(calcBuf3, TILE_LEN * sizeof(uint32_t));
        pipe->InitBuffer(calcBuf4, TILE_LEN * sizeof(uint32_t));
        LocalTensor<uint32_t> tempLocal0 = calcBuf0.Get<uint32_t>();
        LocalTensor<uint32_t> tempLocal1 = calcBuf1.Get<uint32_t>();
        LocalTensor<uint32_t> tempLocal2 = calcBuf2.Get<uint32_t>();
        LocalTensor<uint32_t> tempLocal3 = calcBuf3.Get<uint32_t>();
        LocalTensor<uint32_t> tempLocal4 = calcBuf4.Get<uint32_t>();

        pipe->InitBuffer(max_bits_length, TILE_NUM * sizeof(uint8_t));
        LocalTensor<uint8_t> mblLocal = max_bits_length.Get<uint8_t>();
        //每个tile处理两个，对应两个max_bit_length = 1字节

        pipe->InitBuffer(out, DATA_BLOCK_BYTE_NUM / 2);
        LocalTensor<uint16_t> e_outLocal = out.Get<uint16_t>();

        pipe->InitBuffer(prefixBuf, TILE_NUM * sizeof(uint32_t));
        LocalTensor<uint32_t> prefixLocal = prefixBuf.Get<uint32_t>();

        for(uint32_t i = blockId; i < datablockNum; i += blockNum){
            uint32_t offset0 = i * DATA_BLOCK_BYTE_NUM / sizeof(uint32_t);
            // int32_t end = 0;
            // if(offset0 + DATA_BLOCK_BYTE_NUM < totalUncompressedBytes){
            //     end = (int)offset0 + DATA_BLOCK_BYTE_NUM;
            // }
            // else{
            //     end = (int)totalUncompressedBytes;
            // }
            // int end = std::min((int)offset0 + DATA_BLOCK_BYTE_NUM, (int)totalUncompressedBytes);
            // this->blockDataBytesSize = end - offset0;
            compressedsize = 0;// 字节为单位
            for(uint32_t tileIdx = 0; tileIdx < TILE_NUM; ++tileIdx){
                uint32_t offset1 = tileIdx * TILE_LEN;
                CopyIn(offset0 + offset1);//输入队列每次都copy32字节
                Compute(tileIdx, compressedsize, mblLocal, e_outLocal, tableLocal, blLocal, tempLocal0, tempLocal1, tempLocal2, tempLocal3, maskLocal, prefixLocal);
            }
            // compressedSize = compressedsize;
            // compSize(i) = 4096;
            // assert(compressedsize == 2048);
            compressedsize = ((compressedsize + 32 - 1) / 32) * 32;// 向上取到32的倍数
            DataCopy(tileprefix[i * TILE_NUM], prefixLocal, TILE_NUM);
            DataCopy(output[i * DATA_BLOCK_BYTE_NUM / 2 / sizeof(uint16_t)], e_outLocal, compressedsize / sizeof(uint16_t));// 注意：output是uint16_t
            DataCopy(mbl_output[i * TILE_NUM], mblLocal, TILE_NUM);// 每次DataCopy的数据是32字节的倍数
            // if(i == 0)
            // assert(e_outLocal(826) == 0);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t offset) {
        LocalTensor<T> inLocal = inQueue.AllocTensor<T>();
        // 拷贝当前Tile数据到Local
        DataCopy(inLocal, e_input[offset], TILE_LEN);
        inQueue.EnQue(inLocal);
    }
/*
    __aicore__ inline void Merge_Vec(uint32_t max_bits_length, LocalTensor<uint32_t>& encodedData, LocalTensor<uint16_t>& e_outLocal, LocalTensor<uint32_t>& mergeLocal0, LocalTensor<uint32_t>& mergeLocal1, LocalTensor<uint32_t>& mergeLocal2){
        //达到16bit就写出到e_outLocal
        if(max_bits_length == 0){// 最大截断bit = 0，直接不保存
            return;
        }
        else if(max_bits_length == 1){
            ShiftLeft(mergeLocal0, encodedData, (uint32_t)1, TILE_LEN);
            Or(mergeLocal1, encodedData, mergeLocal0[1], TILE_LEN * 2);//长度为2

            ShiftLeft(mergeLocal0, mergeLocal1, (uint32_t)2, TILE_LEN);
            Or(mergeLocal2, mergeLocal1, mergeLocal0[2], TILE_LEN);//长度为4

            ShiftLeft(mergeLocal0, mergeLocal2, (uint32_t)4, TILE_LEN);
            Or(mergeLocal1, mergeLocal2, mergeLocal0[4], TILE_LEN);//长度为8

            ShiftLeft(mergeLocal0, mergeLocal1, (uint32_t)8, TILE_LEN);
            Or(mergeLocal2, mergeLocal1, mergeLocal0[8], TILE_LEN);//长度为16

            e_outLocal(0) = (uint16_t)mergeLocal2(0);
            e_outLocal(1) = (uint16_t)mergeLocal2(16);
        }
        else if(max_bits_length == 2){
            ShiftLeft(mergeLocal0, encodedData, (uint32_t)2, TILE_LEN);
            Or(mergeLocal1, encodedData, mergeLocal0[1], TILE_LEN * 2);//长度为4

            ShiftLeft(mergeLocal0, mergeLocal1, (uint32_t)4, TILE_LEN);
            Or(mergeLocal2, mergeLocal1, mergeLocal0[2], TILE_LEN * 2);//长度为8

            ShiftLeft(mergeLocal0, mergeLocal2, (uint32_t)8, TILE_LEN);
            Or(mergeLocal1, mergeLocal2, mergeLocal0[4], TIEL_LEN * 2);//长度为16

            e_outLocal(0) = (uint16_t)mergeLocal1(0);
            e_outLocal(1) = (uint16_t)mergeLocal1(8);
            e_outLocal(2) = (uint16_t)mergeLocal1(16);
            e_outLocal(3) = (uint16_t)mergeLocal1(24);
        }
        else if(max_bits_length == 3){
            ShiftLeft(mergeLocal0, encodedData, (uint32_t)3, TIEL_LEN);
            Or(mergeLocal1, encodedData, mergeLocal0[1], TILE_LEN * 2);//长度为6

            ShiftLeft(mergeLocal0, mergeLocal1, (uint32_t)6, TILE_LEN);
            Or(mergeLocal2, mergeLocal1, mergeLocal0[2], TILE_LEN * 2);//长度为12

            ShiftLeft(mergeLocal0, mergeLocal2, (uint32_t)9, TILE_LEN);
            Or(mergeLocal1, mergeLocal2, mergeLocal0[4], TILE_LEN * 2);//长度为24

            e_outLocal(0) = (uint16_t)mergeLocal1(0);
            e_outLocal(1) = (uint16_t)mergeLocal1(8);
            e_outLocal(2) = (uint16_t)mergeLocal1(16);
            e_outLocal(3) = (uint16_t)mergeLocal1(24);

            ShiftRight(mergeLocal0, mergeLocal1, (uint32_t)16, TILE_LEN);
            ShiftLeft(mergeLocal2, mergeLocal0, (uint32_t)8, TILE_LEN);
            Or(mergeLocal1, mergeLocal0, mergeLocal2[8], TILE_LEN * 2);

            e_outLocal(4) = (uint16_t)mergeLocal1(0);
            e_outLocal(5) = (uint16_t)mergeLocal1(16);
        }
        else if(max_bits_length == 4){
            ShiftLeft(mergeLocal0, encodedData, (uint32_t)4, TILE_LEN);
            Or(mergeLocal1, encodedData, mergeLocal0[1], TILE_LEN * 2);//长度为8

            ShiftLeft(mergeLocal0, mergeLocal1, (uint32_t)8, TILE_LEN);
            Or(mergeLocal2, mergeLocal1, mergeLocal0[2], TILE_LEN * 2);//长度为16

            e_outLocal(0) = (uint16_t)mergeLocal2(0);
            e_outLocal(1) = (uint16_t)mergeLocal2(4);
            e_outLocal(2) = (uint16_t)mergeLocal2(8);
            e_outLocal(3) = (uint16_t)mergeLocal2(12);
            e_outLocal(4) = (uint16_t)mergeLocal2(16);
            e_outLocal(5) = (uint16_t)mergeLocal2(20);
            e_outLocal(6) = (uint16_t)mergeLocal2(24);
            e_outLocal(7) = (uint16_t)mergeLocal2(28);
        }
        else if(max_bits_length == 5){
            ShiftLeft(mergeLocal0, encodedData, (uint32_t)5, TILE_LEN);
            Or(mergeLocal1, encodedData, mergeLocal0[1], TILE_LEN * 2);//长度为10

            ShiftLeft(mergeLocal0, mergeLocal1, (uint32_t)10, TILE_LEN);
            Or(mergeLocal2, mergeLocal1, mergeLocal0[2], TILE_LEN * 2);//长度为20

            e_outLocal(0) = (uint16_t)mergeLocal2(0);
            e_outLocal(1) = (uint16_t)mergeLocal2(4);
            e_outLocal(2) = (uint16_t)mergeLocal2(8);
            e_outLocal(3) = (uint16_t)mergeLocal2(12);
            e_outLocal(4) = (uint16_t)mergeLocal2(16);
            e_outLocal(5) = (uint16_t)mergeLocal2(20);
            e_outLocal(6) = (uint16_t)mergeLocal2(24);
            e_outLocal(7) = (uint16_t)mergeLocal2(28);

            ShiftRight(mergeLocal0, mergeLocal2, (uint32_t)16, TILE_LEN);
            ShiftLeft(mergeLocal1, mergeLocal0, (uint16_t)4, TILE_LEN);
            Or(mergeLocal2, mergeLocal0, mergeLocal1[4], TILE_LEN * 2);//长度为8

            ShiftLeft(mergeLocal1, mergeLocal2, (uint32_t)8, TILE_LEN);
            Or(mergeLocal0, mergeLocal2, mergeLocal1[8], TILE_LEN * 2);//长度为16

            e_outLocal(8) = (uint16_t)mergeLocal0(0);
            e_outLocal(9) = (uint16_t)mergeLocal0(16);
        }
        else if(max_bits_length == 6){
            ShiftLeft(mergeLocal0, encodedData, (uint32_t)6, TILE_LEN);
            Or(mergeLocal1, encodedData, mergeLocal0[1], TILE_LEN * 2);//长度为12

            ShiftLeft(mergeLocal0, mergeLocal1, (uint32_t)12, TILE_LEN);
            Or(mergeLocal2, mergeLocal1, mergeLocal0[2], TILE_LEN * 2);//长度为24

            e_outLocal(0) = (uint16_t)mergeLocal2(0);
            e_outLocal(1) = (uint16_t)mergeLocal2(4);
            e_outLocal(2) = (uint16_t)mergeLocal2(8);
            e_outLocal(3) = (uint16_t)mergeLocal2(12);
            e_outLocal(4) = (uint16_t)mergeLocal2(16);
            e_outLocal(5) = (uint16_t)mergeLocal2(20);
            e_outLocal(6) = (uint16_t)mergeLocal2(24);
            e_outLocal(7) = (uint16_t)mergeLocal2(28);

            ShiftRight(mergeLocal0, mergeLocal2, (uint32_t)16, TILE_LEN);
            ShiftLeft(mergeLocal1, mergeLocal0, (uint16_t)4, TILE_LEN);
            Or(mergeLocal2, mergeLocal0, mergeLocal1[4], TILE_LEN * 2);//长度为16

            e_outLocal(8) = (uint16_t)mergeLocal2(0);
            e_outLocal(9) = (uint16_t)mergeLocal2(8);
            e_outLocal(10) = (uint16_t)mergeLocal2(16);
            e_outLocal(11) = (uint16_t)mergeLocal2(24);
        }
        else if(max_bits_length == 7){
            ShiftLeft(mergeLocal0, encodedData, (uint32_t)7, TILE_LEN);
            Or(mergeLocal1, encodedData, mergeLocal0[1], TILE_LEN * 2);//长度为14

            ShiftLeft(mergeLocal0, mergeLocal1, (uint32_t)14, TILE_LEN);
            Or(mergeLocal2, mergeLocal1, mergeLocal0[2], TILE_LEN * 2);//长度为28

            e_outLocal(0) = (uint16_t)mergeLocal2(0);
            e_outLocal(1) = (uint16_t)mergeLocal2(4);
            e_outLocal(2) = (uint16_t)mergeLocal2(8);
            e_outLocal(3) = (uint16_t)mergeLocal2(12);
            e_outLocal(4) = (uint16_t)mergeLocal2(16);
            e_outLocal(5) = (uint16_t)mergeLocal2(20);
            e_outLocal(6) = (uint16_t)mergeLocal2(24);
            e_outLocal(7) = (uint16_t)mergeLocal2(28);

            ShiftRight(mergeLocal0, mergeLocal2, (uint32_t)16, TILE_LEN);
            ShiftLeft(mergeLocal1, mergeLocal0, (uint16_t)4, TILE_LEN);
            Or(mergeLocal2, mergeLocal0, mergeLocal1[4], TILE_LEN * 2);//长度为24

            e_outLocal(8) = (uint16_t)mergeLocal2(0);
            e_outLocal(9) = (uint16_t)mergeLocal2(8);
            e_outLocal(10) = (uint16_t)mergeLocal2(16);
            e_outLocal(11) = (uint16_t)mergeLocal2(24);

            ShiftRight(mergeLocal0, mergeLocal2, (uint32_t)16, TILE_LEN);
            ShiftLeft(mergeLocal1, mergeLocal0, (uint16_t)4, TILE_LEN);
            Or(mergeLocal2, mergeLocal0, mergeLocal1[8], TILE_LEN * 2);//长度为16

            e_outLocal(12) = (uint16_t)mergeLocal2(0);
            e_outLocal(13) = (uint16_t)mergeLocal2(16);  
        }
        else if(max_bits_length == 8){
            ShiftLeft(mergeLocal0, encodedData, (uint32_t)8, TILE_LEN);
            Or(mergeLocal1, encodedData, mergeLocal0[1], TILE_LEN * 2);//长度为16

            e_outLocal(0) = (uint16_t)mergeLocal1(0);
            e_outLocal(1) = (uint16_t)mergeLocal1(2);
            e_outLocal(2) = (uint16_t)mergeLocal1(4);
            e_outLocal(3) = (uint16_t)mergeLocal1(6);
            e_outLocal(4) = (uint16_t)mergeLocal1(8);
            e_outLocal(5) = (uint16_t)mergeLocal1(10);
            e_outLocal(6) = (uint16_t)mergeLocal1(12);
            e_outLocal(7) = (uint16_t)mergeLocal1(14);
            e_outLocal(8) = (uint16_t)mergeLocal1(16);
            e_outLocal(9) = (uint16_t)mergeLocal1(18);
            e_outLocal(10) = (uint16_t)mergeLocal1(20);
            e_outLocal(11) = (uint16_t)mergeLocal1(22);
            e_outLocal(12) = (uint16_t)mergeLocal1(24);
            e_outLocal(13) = (uint16_t)mergeLocal1(26);
            e_outLocal(14) = (uint16_t)mergeLocal1(28);
            e_outLocal(15) = (uint16_t)mergeLocal1(30);
        }
    }
*/
    __aicore__ inline void Merge(uint32_t index, uint32_t max_bits_length, LocalTensor<uint32_t>& encodedData, LocalTensor<uint16_t>& e_outLocal){
        //达到16bit就写出到e_outLocal
        if(max_bits_length == 0){// 最大截断bit = 0，直接不保存
            return;
        }
        uint32_t buffer = 0;
        uint32_t bit_shift = 0;
        for(int i = 0; i < TILE_LEN; i ++){
            int num = ((uint32_t)encodedData(i)) << bit_shift;
            buffer |= num;
            bit_shift += max_bits_length;
            if(bit_shift >= 16){
                e_outLocal(index) = (uint16_t)buffer;
                index ++;
                buffer >>= 16;
                bit_shift -= 16;
            }
        }
    }

    __aicore__ inline void Compute(
                                   uint32_t tileIdx,
                                   uint32_t& compressedsize,
                                   LocalTensor<uint8_t>& mblLocal,
                                   LocalTensor<uint16_t>& e_outLocal,
                                   LocalTensor<T>& tableLocal,
                                   LocalTensor<T>& blLocal,
                                   LocalTensor<T>& tempLocal0,
                                   LocalTensor<T>& tempLocal1,
                                   LocalTensor<T>& tempLocal2,
                                   LocalTensor<T>& tempLocal3,
                                //    LocalTensor<T>& mergeLocal0,
                                //    LocalTensor<T>& mergeLocal1,
                                //    LocalTensor<T>& mergeLocal2,
                                   LocalTensor<T>& maskLocal,
                                   LocalTensor<T>& prefixLocal
    ) {

        LocalTensor<T> e_inLocal = inQueue.DeQue<T>();
        // if(tileIdx == 0)
        // assert(e_inLocal(31) == 14745627);

        And(tempLocal0, e_inLocal, maskLocal, (int32_t)TILE_LEN * 2);
        ShiftLeft(tempLocal3, tempLocal0, (uint32_t)2, TILE_LEN);//左移2位相当于乘sizeof(uint32_t)，因为gather偏置是字节为单位
        Gather(tempLocal1, tableLocal, tempLocal3, (uint32_t)0, (uint32_t)TILE_LEN);//gather编码表，为最终用于编码紧缩的部分
        ShiftLeft(tempLocal3, tempLocal1, (uint32_t)2, TILE_LEN);//左移2位相当于乘sizeof(uint32_t)，因为gather偏置是字节为单位，必须与T的位宽对齐，不然出现未知错误
        Gather(tempLocal2, blLocal, tempLocal3, (uint32_t)0, (uint32_t)TILE_LEN);//gather比特长度
        ReduceMax(tempLocal1.template ReinterpretCast<float>() , tempLocal2.template ReinterpretCast<float>() , tempLocal0.template ReinterpretCast<float>() , TILE_LEN, 0);
        //求出最大截断bits长度，归约操作
        int32_t max_bits_length0 = tempLocal1(0);
        //0;
        // for(int i = 0; i < TILE_LEN; i ++){ 
        //     int num = tempLocal2(i);
        //     if(num > max_bits_length0){
        //         max_bits_length0 = num;
        //     }
        // }
        Merge(compressedsize / sizeof(uint16_t), max_bits_length0, tempLocal1, e_outLocal);
        compressedsize = compressedsize + (max_bits_length0 * (TILE_LEN / 8));

        ShiftRight(tempLocal0, e_inLocal, (uint32_t)16, TILE_LEN);
        ShiftLeft(tempLocal3, tempLocal0, (uint32_t)2, TILE_LEN);//左移2位相当于乘sizeof(uint32_t)，因为gather偏置是字节为单位
        Gather(tempLocal1, tableLocal, tempLocal3, (uint32_t)0, TILE_LEN);//gather编码表
        ShiftLeft(tempLocal3, tempLocal1, (uint32_t)2, TILE_LEN);//左移2位相当于乘sizeof(uint32_t)，因为gather偏置是字节为单位
        Gather(tempLocal2, blLocal, tempLocal3, (uint32_t)0, TILE_LEN);//gather比特长度
        ReduceMax(tempLocal1.template ReinterpretCast<float>() , tempLocal2.template ReinterpretCast<float>() , tempLocal0.template ReinterpretCast<float>() , TILE_LEN, 0);
        int32_t max_bits_length1 = tempLocal1(0);
        // 0;
        // for(int i = 0; i < TILE_LEN; i ++){
        //     int num = tempLocal2(i);
        //     if(num > max_bits_length1){
        //         max_bits_length1 = num;
        //     }
        // }
        Merge(compressedsize / sizeof(uint16_t), max_bits_length1, tempLocal1, e_outLocal);
        compressedsize = compressedsize + (max_bits_length1 * (TILE_LEN / 8));
        prefixLocal(tileIdx) = compressedsize;

        mblLocal(tileIdx) = (max_bits_length1 << 4) | max_bits_length0;

        inQueue.FreeTensor(e_inLocal);
    }

private:
    TPipe* pipe;
    TQue<QuePosition::VECIN, 1> inQueue;
    // TQue<QuePosition::VECOUT, 1> e_outQueue;

    TBuf<AscendC::TPosition::VECCALC> table;
    TBuf<AscendC::TPosition::VECCALC> bits_length;
    TBuf<AscendC::TPosition::VECCALC> calcBuf0;
    TBuf<AscendC::TPosition::VECCALC> calcBuf1;
    TBuf<AscendC::TPosition::VECCALC> calcBuf2;
    TBuf<AscendC::TPosition::VECCALC> calcBuf3;
    TBuf<AscendC::TPosition::VECCALC> calcBuf4;
    TBuf<AscendC::TPosition::VECCALC> max_bits_length;
    TBuf<AscendC::TPosition::VECCALC> out;
    TBuf<AscendC::TPosition::VECCALC> writeBuf0;
    TBuf<AscendC::TPosition::VECCALC> writeBuf1;
    TBuf<AscendC::TPosition::VECCALC> writeBuf2;
    TBuf<TPosition::VECCALC> mask;
    TBuf<TPosition::VECCALC> prefixBuf;

    GlobalTensor<T> e_input;
    GlobalTensor<T> table_input;
    GlobalTensor<uint32_t> bits_length_input;
    GlobalTensor<uint8_t> mbl_output;
    GlobalTensor<uint16_t> output;
    GlobalTensor<uint32_t> tileprefix;

    uint32_t blockDataBytesSize;
    uint32_t datablockNum;
    uint32_t blockId;
    uint32_t blockNum;
    uint32_t compressedsize;//当前压缩后的字节数
    uint32_t totalUncompressedBytes;
};

// template<typename T>// T =int32_t
// class PrefixKernel {
// public:
//     __aicore__ inline PrefixKernel() {}// 计算独占前缀和

//     __aicore__ inline void Init(TPipe* pipe,
//                                 uint32_t datablockNum,
//                                 __gm__ uint8_t* tilePrefix,
//                                 __gm__ uint8_t* compressedSize,// 输入
//                                 __gm__ uint8_t* compressedSizePrefix// 输出
                                
                                
//     ) {
//         this->pipe = pipe;
//         this->DATA_BLOCK_NUM = datablockNum;
//         tileprefix.SetGlobalBuffer((__gm__ T*)(tilePrefix));
//         compSize.SetGlobalBuffer((__gm__ T*)(compressedSize));
//         output.SetGlobalBuffer((__gm__ T*)(compressedSizePrefix));

//         pipe->InitBuffer(inQueue, BUFFER_NUM, TILE_NUM * sizeof(T));
//         pipe->InitBuffer(outQueue, BUFFER_NUM, ((DATA_BLOCK_NUM + 31) / 32) * 32 * sizeof(T));
//     }

//     __aicore__ inline void Process() {
//         pipe->InitBuffer(prefixTemp, ((DATA_BLOCK_NUM + 31) / 32) * 32 * sizeof(T));
//         LocalTensor<T> prefixLocal = prefixTemp.Get<T>();

//         for(int i = 0; i < DATA_BLOCK_NUM; i ++){
//             int offset0 = i * TILE_NUM;
//             CopyIn(offset0);
//             Compute(i, prefixLocal);
//         }
//         // assert();
//         ComputePrefix(prefixLocal);
//         CopyOut(prefixLocal);
//     }

// private:
//     __aicore__ inline void CopyIn(int32_t offset) {
//         LocalTensor<T> inLocal = inQueue.AllocTensor<T>();
//         DataCopy(inLocal, tileprefix[offset], TILE_NUM);
//         inQueue.EnQue(inLocal);
//     }

//     __aicore__ inline void Compute(uint32_t i, LocalTensor<int32_t> prefixLocal) {
//         LocalTensor<T> inLocal = inQueue.DeQue<T>();
//         prefixLocal(i) = 
//         // inLocal(TILE_NUM - 1);
//         ((inLocal(TILE_NUM - 1) + 32 - 1) / 32) * 32;
//         // if(i == 1)
//         // {
//         //     assert(prefixLocal(0) == 2048);
//         // }
//         inQueue.FreeTensor(inLocal);
//     }

//     __aicore__ inline void ComputePrefix(LocalTensor<int32_t> prefixLocal){
//         LocalTensor<T> outLocal = outQueue.AllocTensor<T>();
//         outLocal(0) = 0;
//         for(int l = 1; l < DATA_BLOCK_NUM; l ++){
//             outLocal(l) = outLocal(l - 1) + prefixLocal(l - 1);
//         }
//         // assert(outLocal(0) == 0);
//         // assert(DATA_BLOCK_NUM == 32);
//         outQueue.EnQue(outLocal);
//     }

//     __aicore__ inline void CopyOut(LocalTensor<int32_t> prefixLocal) {
//         LocalTensor<T> outLocal = outQueue.DeQue<T>();
//         DataCopy(output, outLocal, ((DATA_BLOCK_NUM + 31) / 32) * 32);//向上取到32的倍数
//         DataCopy(compSize, prefixLocal, ((DATA_BLOCK_NUM + 31) / 32) * 32);
//         outQueue.FreeTensor(outLocal);
//     }

// private:
//     TPipe* pipe;
//     TQue<QuePosition::VECIN, 1> inQueue;
//     TQue<QuePosition::VECOUT, 1> outQueue;
//     TBuf<TPosition::VECCALC> prefixTemp;

//     GlobalTensor<T> tileprefix;
//     GlobalTensor<T> compSize;
//     GlobalTensor<T> output;

//     uint32_t DATA_BLOCK_NUM;
// };

// template<typename T>
// class CoalesceKernel {
// public:
//     __aicore__ inline CoalesceKernel() {} // 生成数据头，紧缩码字，计算压缩率
//     // 输入：max_bit_length数组（3bits * block_num），码字（max_bit_length * blockSize）
//     // 输出：一整块连续的压缩块，压缩块的大小

//     __aicore__ inline void Init(TPipe* pipe,
//                                 uint32_t dataBlockNum,
//                                 __gm__ uint8_t* finalCompressedExp, //output
//                                 __gm__ uint8_t* compressedSize,
//                                 __gm__ uint8_t* compressedSizePrefix,
//                                 uint32_t totalUncompressedBytes) {
//         this->pipe = pipe;
//         this->dataBlockNum = dataBlockNum;
//         this->blockId = GetBlockIdx();
//         this->blockNum = GetBlockNum();

//         input.SetGlobalBuffer((__gm__ T*)(finalCompressedExp + DATA_BLOCK_BYTE_NUM / 2 * dataBlockNum));
//         output.SetGlobalBuffer((__gm__ T*)(finalCompressedExp));
//         compressedsize.SetGlobalBuffer((__gm__ T*)(compressedSize));
//         compressedsizePrefix.SetGlobalBuffer((__gm__ T*)(compressedSizePrefix));

//         pipe->InitBuffer(queBind, BUFFER_NUM, DATA_BLOCK_BYTE_NUM / 2);
//     }

// public:
//     __aicore__ inline void Process()
//     {
//         pipe->InitBuffer(compSize, dataBlockNum * sizeof(T));
//         LocalTensor<T> compSizeLocal = compSize.Get<T>();
//         DataCopy(compSizeLocal, compressedsize, dataBlockNum);

//         pipe->InitBuffer(compPrefix, dataBlockNum * sizeof(T));
//         LocalTensor<T> compPrefixLocal = compPrefix.Get<T>();
//         DataCopy(compPrefixLocal, compressedsizePrefix, dataBlockNum);

//         // pipe->InitBuffer(copy, DATA_BLOCK_BYTE_NUM / 2);
//         // LocalTensor<T> bindLocal = copy.Get<T>();
//         auto bindLocal = queBind.AllocTensor<T>();
//         for(int i = blockId; i < dataBlockNum; i += blockNum){
//             uint32_t compSize = compSizeLocal(i);
//             uint32_t compSizePrefix = compPrefixLocal(i);//字节为单位
//             DataCopy(bindLocal, input[i * DATA_BLOCK_BYTE_NUM / 2 / sizeof(T)], compSize / sizeof(T));
//             // CopyIn(i);
//             // CopyOut(i);
//             DataCopy(output[compSizePrefix / sizeof(T)], bindLocal, compSize / sizeof(T));
//             // if(i == 0)
//             // assert(compSizePrefix == 0);
//             // if(i == 1)
//             // assert(compSizePrefix == 1664);
//         }
//         queBind.FreeTensor(bindLocal);

//     }
// // private:
// //     __aicore__ inline void CopyIn(int i){
// //         auto bindLocal = queBind.AllocTensor<T>();
// //         DataCopy(bindLocal, input[i * 2048 / sizeof(T)], compressedsize(i) / sizeof(T));
// //         queBind.EnQue(bindLocal);
// //         // queBind.FreeTensor(bindLocal);
// //     }
// //     __aicore__ inline void CopyOut(int i){
// //         auto bindLocal = queBind.DeQue();
// //         DataCopy(output[compressedsizePrefix(i)], bindLocal, compressedsize(i) / sizeof(T));
// //         // queBind.FreeTensor(bindLocal);
// //     }
// private:
//     TPipe* pipe;
//     TQueBind<TPosition::VECIN, TPosition::VECOUT, BUFFER_NUM> queBind;
//     TBuf<TPosition::VECCALC> copy;
//     TBuf<TPosition::VECCALC> compSize;
//     TBuf<TPosition::VECCALC> compPrefix;

//     GlobalTensor<T> input;//输入每个数据块压缩后的GM地址
//     GlobalTensor<T> output;//输出每个数据块压缩后的GM地址
//     GlobalTensor<T> compressedsize;
//     GlobalTensor<T> compressedsizePrefix;

//     uint32_t blockId;
//     uint32_t blockNum;
//     uint32_t dataBlockNum;
// };

// __global__ __aicore__ void extractbits_and_histogram(
//                                 uint32_t datablockNum,//数据块数量
//                                 __gm__ uint8_t* in, 
//                                 __gm__ uint8_t* tempBuffer, 
//                                 __gm__ uint8_t* final, 
//                                 __gm__ uint8_t* histogramDevice, 
//                                 uint32_t totalUncompressedSize)
// {
//     TPipe pipe;
//     Extractbits_and_histogramKernel<uint32_t> op;
//     op.Init(&pipe, datablockNum, in, tempBuffer, final, histogramDevice, totalUncompressedSize);
//     op.Process();
// }

// __global__ __aicore__ void MergeHistogram(__gm__ uint8_t* hist_in,
//                                           __gm__ uint8_t* table)
// {
//     TPipe pipe;
//     MergeHistogramKernel<int32_t> op;
//     op.Init(&pipe, hist_in, table);
//     op.Process();
// }

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

// __global__ __aicore__ void calcprefix(uint32_t datablockNum,
//                                       __gm__ uint8_t* tilePrefix,
//                                       __gm__ uint8_t* compressedSize,// 输入
//                                       __gm__ uint8_t* compressedSizePrefix
//                                       )
// {
//     TPipe pipe;
//     PrefixKernel<int32_t> op;
//     op.Init(&pipe, datablockNum, tilePrefix, compressedSize, compressedSizePrefix);
//     op.Process();
// }

// __global__ __aicore__ void coalesce(uint32_t dataBlockNum,
//                                     __gm__ uint8_t* finalCompressedExp, //output
//                                     __gm__ uint8_t* compressedSize,
//                                     __gm__ uint8_t* compressedSizePrefix,
//                                     uint32_t totalUncompressedBytes)
// {
//     TPipe pipe;
//     CoalesceKernel<uint32_t> op;
//     op.Init(&pipe, dataBlockNum, finalCompressedExp, compressedSize, compressedSizePrefix, totalUncompressedBytes);
//     op.Process();
// }

extern "C" void compress(uint32_t datablockNum, void* stream, uint8_t* srcDevice, uint8_t* tempBuffer, uint8_t* final, int32_t* histogramDevice, uint32_t* tilePrefix, uint32_t* compressedSize, uint32_t* compressedSizePrefix, uint32_t totalUncompressedSize) {
    // extractbits_and_histogram<<<BLOCK_NUM, nullptr, stream>>>(datablockNum, srcDevice, tempBuffer, final, histogramDevice, totalUncompressedSize);//提取字节并计算直方图
    // MergeHistogram<<<1, nullptr, stream>>>(reinterpret_cast<uint8_t*>(histogramDevice), final + 32);
    comp<<<BLOCK_NUM, nullptr, stream>>>(datablockNum, tempBuffer, final, reinterpret_cast<uint8_t*>(histogramDevice), reinterpret_cast<uint8_t*>(tilePrefix), totalUncompressedSize);//压缩函数
    // calcprefix<<<1, nullptr, stream>>>(datablockNum, reinterpret_cast<uint8_t*>(tilePrefix), reinterpret_cast<uint8_t*>(compressedSize), reinterpret_cast<uint8_t*>(compressedSizePrefix));//计算前缀和，用于后续块合并，字节为单位，
    // coalesce<<<BLOCK_NUM, nullptr, stream>>>(datablockNum, final + 32 + HISTOGRAM_BINS * sizeof(uint8_t) + TILE_NUM * datablockNum + DATA_BLOCK_BYTE_NUM / 2 * datablockNum, reinterpret_cast<uint8_t*>(compressedSize), reinterpret_cast<uint8_t*>(compressedSizePrefix), totalUncompressedSize);//纯搬运内核
}


