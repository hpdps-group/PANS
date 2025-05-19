#include "kernel_operator.h"
#include "hans_utils.h"

using namespace AscendC;

constexpr uint32_t DATA_BLOCK_BYTE_NUM = 4096;
constexpr uint32_t DATA_BLOCK_NUM = ?;
constexpr int32_t BUFFER_NUM = 2; 
constexpr int32_t BLOCK_NUM = 256;
constexpr uint32_t HISTOGRAM_BINS = 256;// 尽可能是2的幂
constexpr uint32_t HANDLE_NUM_PER = 32; // 算子每次向量化处理32单位的数据量，直方图计算中每个block生成32个temp_table，一共32 * 4 * 256 = 32KB
constexpr uint32_t TILE_LEN = 32; // 每个Tile处理32个单元(单元指输入数据的类型)
constexpr uint32_t TILE_NUM = 32; // 每个数据块包含TILE_NUM个TILE

//注意：所有算子的输入与输出尽可能32字节对齐，Add这些底层接口的输入与输出必须32字节对齐

template<typename T>
class Extractbits_and_histogramKernel {
public:
    __aicore__ inline Extractbits_and_histogramKernel() {} // 切分数据，分离指数位，同时进行histogram统计
    // 输入：uint16_t数组(两两组成一个int32_t)
    // 输出：指数数组（int32_t），尾数+sign数组（uint8_t)，histogram统计数组（int32_t）

    __aicore__ inline void Init(Tpipe& pipe,
                                __gm__ uint8_t* in, 
                                __gm__ uint8_t* e_out, 
                                __gm__ uint8_t* m_s_out, 
                                __gm__ uint8_t* hist_out, 
                                uint32_t totalElements) {
        this->pipe = pipe;
        uint32_t total = *(reinterpret_cast<const uint32_t*>(totalElements));
        uint32_t blockId = GetBlockIdx();
        uint32_t blockNum = GetBlockNum();
        uint32_t perBlock = (total + blockNum - 1) / blockNum;
        uint32_t start = blockId * perBlock;
        if (start >= total) return;
        uint32_t end = min(start + perBlock, total);
        this->blockElements = end - start;

        input.SetGlobalBuffer((__gm__ uint32_t*)(in + sizeof(uint32_t) * start));
        e_output.SetGlobalBuffer((__gm__ uint32_t*)(e_out + sizeof(uint32_t) * start));
        m_s_output.SetGlobalBuffer((__gm__ uint32_t*)(m_s_out + sizeof(uint32_t) * start));
        hist_output.SetGlobalBuffer((__gm__ int32_t*)(hist_out + sizeof(int32_t) * HISTOGRAM_BINS * blockId));

        pipe.InitBuffer(inQueue, BUFFER_NUM, TILE_LEN * sizeof(T));
        pipe.InitBuffer(e_outQueue, BUFFER_NUM, TILE_LEN * sizeof(uint32_t));
        pipe.InitBuffer(m_s_outQueue, BUFFER_NUM, TILE_LEN / 2 * sizeof(uint32_t));
        //因为开启了double_buffer，最多只能开四个queue
    }

    __aicore__ inline void Process() {

        pipe.InitBuffer(calcBuf0, TILE_LEN * sizeof(uint32_t));
        pipe.InitBuffer(calcBuf1, TILE_LEN * sizeof(uint32_t));
        pipe.InitBuffer(calcBuf2, TILE_LEN * sizeof(uint32_t));
        pipe.InitBuffer(calcBuf3, TILE_LEN * sizeof(uint32_t));
        pipe.InitBuffer(tempHist, TILE_LEN * HISTOGRAM_BINS * sizeof(int32_t));
        pipe.InitBuffer(histBuffer0, TILE_LEN * sizeof(int32_t));
        pipe.InitBuffer(histBuffer1, TILE_LEN * sizeof(int32_t));
        pipe.InitBuffer(mask0, TILE_LEN * sizeof(uint32_t));
        pipe.InitBuffer(mask1, TILE_LEN * sizeof(uint32_t));
        pipe.InitBuffer(mask2, TILE_LEN * sizeof(uint32_t));
        pipe.InitBuffer(mask3, TILE_LEN * sizeof(uint32_t));
        pipe.InitBuffer(mask4, TILE_LEN * sizeof(uint32_t));
        pipe.InitBuffer(offsetBuffer, TILE_LEN * sizeof(int32_t));
        pipe.InitBuffer(one, TILE_LEN * sizeof(int32_t));

        LocalTensor<uint32_t> tempLocal0 = calcBuf0.Get<uint32_t>();
        LocalTensor<uint32_t> tempLocal1 = calcBuf1.Get<uint32_t>();
        LocalTensor<uint32_t> tempLocal2 = calcBuf2.Get<uint32_t>();
        LocalTensor<uint32_t> tempLocal3 = calcBuf3.Get<uint32_t>();
        LocalTensor<int32_t> histogram = tempHist.Get<int32_t>();
        LocalTensor<int32_t> histTensor0 = histBuffer0.Get<int32_t>();
        LocalTensor<int32_t> histTensor1 = histBuffer1.Get<int32_t>();
        LocalTensor<T> mask0_tensor = mask0.Get<T>();
        LocalTensor<T> mask1_tensor = mask1.Get<T>();
        LocalTensor<T> mask2_tensor = mask2.Get<T>();
        LocalTensor<T> mask3_tensor = mask3.Get<T>();
        LocalTensor<T> mask4_tensor = mask4.Get<T>();
        LocalTensor<int32_t> all_one = one.Get<int32_t>();
        LocalTensor<int32_t> offset_tensor = offset.Get<int32_t>();

        Duplicate(histogram, (int32_t)0, HISTOGRAM_BINS);// 初始化全0
        Duplicate(mask0_tensor, (uint32_t)65280, TILE_LEN);//11111111 00000000
        Duplicate(mask1_tensor, (uint32_t)255, TILE_LEN);//00000000 11111111
        Duplicate(mask2_tensor, (uint32_t)16711935, TILE_LEN);//00000000 11111111 00000000 11111111
        Duplicate(mask3_tensor, (uint32_t)4278255360, TILE_LEN);//11111111 00000000 11111111 00000000
        Duplicate(mask4_tensor, (uint32_t)65535, TILE_LEN);//00000000 00000000 11111111 11111111
        Duplicate(all_one, (int32_t)1, TILE_LEN);//0000 0000 0000 0001
        uint32_t num = ((1 << 16) + 1) << 10;
        for(int i = 0; i < TILE_LEN; i ++){
            offset_tensor(i) = i * num;
        }

        for (uint32_t tileIdx = 0; tileIdx < //1
        tileNum
        ; ++tileIdx) {
            uint32_t offset = tileIdx * TILE_LEN;
            uint32_t len = min((int)TILE_LEN, (int)(blockElements - offset));
            // uint32_t len = TILE_LEN;
            CopyIn(offset, len);
            Compute(len, mask0_tensor, mask1_tensor, mask2_tensor, mask3_tensor, mask4_tensor, all_one, offset_tensor, histogram, tempLocal0, tempLocal1, tempLocal2, tempLocal3, histTensor0, histTensor1);
            CopyOut(offset, len);
        }
        MergeLocalHist(histogram);// 合并TILE_LEN个temp直方图为最终的一个
    }

private:
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t len) {
        LocalTensor<T> inLocal = inQueue.AllocTensor<T>();

        DataCopy(inLocal, input[offset], len);
        inQueue.EnQue(inLocal);
    }

    __aicore__ inline void Compute(uint32_t len,
                                   LocalTensor<T>& mask0_tensor,
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
        LocalTensor<uint32_t> e_outLocal = e_outQueue.AllocTensor<uint32_t>();
        LocalTensor<uint32_t> m_s_outLocal = m_s_outQueue.AllocTensor<uint32_t>();

        // len /= 2;
        // 处理每个元素，每次提取32个int32_t（取出64个uint16_t）
        ShiftLeft(//e_outLocal0,
            tempLocal0, 
            inLocal, (uint32_t)1, len);
        // assert(inLocal.GetValue(0) == 65535);
        ShiftRight(tempLocal1, inLocal, (uint32_t)31, len);//int类型自动算数移位
        // And(tempLocal2, tempLocal1, all_one, len);//提取sign
        Or(//e_outLocal0,
            tempLocal2, 
            tempLocal0, tempLocal1, (int32_t)len * 2);//将sign放在最后

        And(tempLocal0, tempLocal2, mask2_tensor, (int32_t)len * 2);//取出从高到低1和3字节，尾数部分
        ShiftLeft(tempLocal1, tempLocal0[16], (int32_t)8, (int32_t)(len / 2));
        Or(m_s_outLocal, tempLocal0, tempLocal1, (int32_t)len);

        And(tempLocal3, tempLocal2, mask3_tensor, (int32_t)len * 2);//取出从高到低0和2字节，指数部分
        ShiftRight(e_outLocal, tempLocal3, (uint32_t)8, (int32_t)len);//右移8位
        // And(e_outLocal0, tempLocal1, mask1_tensor, (int32_t)len * 2);//取出低8位
        // ShiftRight(tempLocal0, tempLocal1, (uint32_t)16, (int32_t)len);//右移16位
        // And(e_outLocal1, tempLocal0, mask1_tensor, (int32_t)len * 2);//取出低8位
        ShiftRight(tempLocal1, tempLocal3, (uint32_t)6, (int32_t)len);
        Add(tempLocal2.template ReinterpretCast<int32_t>(), tempLocal1.template ReinterpretCast<int32_t>(), offset_tensor, (int32_t)len);

        And(//e_outLocal0,
            tempLocal0, 
            tempLocal2, mask4_tensor, (int32_t)len * 2);//取出低16位
        Gather(histTensor0, histogram, tempLocal0, (uint32_t)0, (uint32_t)len);
        Add(//histogram,
            histTensor1, 
            histTensor0, all_one, (int32_t)len);
        for(int i = 0; i < len; i ++){
            histogram(tempLocal0(i) >> 2) = histTensor1(i);
        }
        // Scatter(histogram.template ReinterpretCast<uint32_t>(), histTensor1.template ReinterpretCast<uint32_t>(), tempLocal0, (uint32_t)0, (uint32_t)len);

        ShiftRight(//e_outLocal1,
            tempLocal0, 
            tempLocal2, (uint32_t)16, (int32_t)len);//取出高16位
        Gather(histTensor0, histogram, tempLocal0, (uint32_t)0, (uint32_t)len);
        Add(//histogram,
            histTensor1, 
            histTensor0, all_one, (int32_t)len);
        // Scatter(histogram, histTensor1, tempLocal0, (uint32_t)0, (uint32_t)len);
        for(int i = 0; i < len; i ++){
            histogram(tempLocal0(i) >> 2) = histTensor1(i);
        }

        inQueue.FreeTensor(inLocal);
        e_outQueue0.EnQue(e_outLocal);
        m_s_outQueue.EnQue(m_s_outLocal);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t len) {
        LocalTensor<uint32_t> e_outLocal = e_outQueue.DeQue<uint32_t>();
        LocalTensor<uint32_t> m_s_outLocal = m_s_outQueue.DeQue<uint32_t>();

        // 将结果拷贝回Global Memory
        DataCopy(e_output0[offset], e_outLocal, len);
        DataCopy(m_s_output[offset], m_s_outLocal, len / 2);

        e_outQueue0.FreeTensor(e_outLocal);
        m_s_outQueue.FreeTensor(m_s_outLocal);
    }

    __aicore__ inline void MergeLocalHist(LocalTensor<int32_t>& histogram) {
        // Add(histogram[0], histogram[0], histogram[16 * HISTOGRAM_BINS], (int32_t)HISTOGRAM_BINS * 16);
        // Add(histogram[0], histogram[0], histogram[8 * HISTOGRAM_BINS], (int32_t)HISTOGRAM_BINS * 8);
        // Add(histogram[0], histogram[0], histogram[4 * HISTOGRAM_BINS], (int32_t)HISTOGRAM_BINS * 4);
        // Add(histogram[0], histogram[0], histogram[2 * HISTOGRAM_BINS], (int32_t)HISTOGRAM_BINS * 2);
        // Add(histogram[0], histogram[0], histogram[1 * HISTOGRAM_BINS], (int32_t)HISTOGRAM_BINS);
        for(int i = 0; i < TILE_NUM; i ++){
            Add(histogram, histogram, histogram[i * HISTOGRAM_BINS], (int32_t)HISTOGRAM_BINS);
        }
        DataCopy(hist_output, histogram, HISTOGRAM_BINS);
    }

private:
    TPipe* pipe;
    TQue<QuePosition::VECIN, 1> inQueue;
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
    GlobalTensor<uint32_t> e_output;
    GlobalTensor<uint32_t> m_s_output;
    GlobalTensor<int32_t> hist_output;

    uint32_t blockElements;//以sizeof(T)为单位的数据量
    uint32_t tileNum;
    uint32_t blockId;
    uint32_t blockNum;
};

template<typename T>
class MergeHistogramKernel {
public:
    __aicore__ inline MergeHistogramKernel() {} // 合并block的直方图
    // 输入：uint16_t数组(两两组成一个int32_t)
    // 输出：指数数组（int32_t），尾数+sign数组（uint8_t)，histogram统计数组（int32_t）

    __aicore__ inline void Init(Tpipe& pipe,
                                __gm__ uint8_t* hist_in) {
        this->pipe = pipe;
        uint32_t blockId = GetBlockIdx();
        uint32_t blockNum = GetBlockNum();

        input.SetGlobalBuffer((__gm__ uint32_t*)(hist_in));
        output.SetGlobalBuffer((__gm__ uint32_t*)(hist_in));

        pipe.InitBuffer(inQueue0, BUFFER_NUM, TILE_LEN * sizeof(T));
        pipe.InitBuffer(inQueue1, BUFFER_NUM, TILE_LEN * sizeof(T));
        pipe.InitBuffer(outQueue, BUFFER_NUM, TILE_LEN * sizeof(uint32_t));
    }

    __aicore__ inline void Process() {
        for(int i = blockNum / 2; i >= 1; i >> 1){
            if(blockId < i){
                CopyIn(i);
                Compute();
                CopyOut();
            }
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t offset) {
        LocalTensor<T> inLocal0 = inQueue0.AllocTensor<T>();
        LocalTensor<T> inLocal1 = inQueue1.AllocTensor<T>();
        DataCopy(inLocal0, hist_out[blockId * HISTOGRAM_BINS], HISTOGRAM_BINS);
        DataCopy(inLocal1, hist_out[(blockId + offset) * HISTOGRAM_BINS], HISTOGRAM_BINS);
        inQueue0.EnQue(inLocal0);
        inQueue1.EnQue(inLocal1);
    }

    __aicore__ inline void Compute(){
        LocalTensor<T> inLocal0 = inQueue0.DeQue<T>();
        LocalTensor<T> inLocal1 = inQueue1.DeQue<T>();
        LocalTensor<T> outLocal = outQueue.DeQue<T>();
        Add(outLocal, inLocal0, inLocal1, (int32_t)HISTOGRAM_BINS);
        outQueue.EnQue<T>(outLocal);
        inQueue.FreeTensor(inLocal);
        outQueue.FreeTensor(outLocal);
    }

    __aicore__ inline void CopyOut(){
        LocalTensor<T> outLocal = outQueue.DeQue<T>();
        DataCopy(outLocal, hist_out[blockId * HISTOGRAM_BINS], HISTOGRAM_BINS);
        DeQue.FreeTensor(outLocal);
    }

private:
    TPipe* pipe;
    TQue<QuePosition::VECIN, 1> inQueue0;
    TQue<QuePosition::VECIN, 1> inQueue1;
    TQue<QuePosition::VECOUT, 1> outQueue;

    GlobalTensor<T> input;
    GlobalTensor<uint32_t> output;

    uint32_t blockId;
    uint32_t blockNum;
};

template<typename T>// int32_t
class CompressKernel {
public:
    __aicore__ inline CompressKernel() {}
    // 输入：指数数组（uint8_t），table(uint8_t)，
    // 输出：max_bit_length数组（3bits * block_num），码字（max_bit_length * blockSize）

    __aicore__ inline void Init(Tpipe& pipe,
                                __gm__ uint8_t* tempBuffer, //e_input
                                __gm__ uint8_t* final, //output
                                __gm__ uint8_t* histogramDevice, //table_input
                                __gm__ uint8_t* bits_lengthDevice, // bits_length数组
                                __gm__ uint8_t* compressedSize, // 用于保存每个数据块最后压缩完的块大小
                                uint32_t totalUncompressedBytes // 保存全部未压缩数据的大小，用于分块拖尾处理
                                ) {
        this->pipe = pipe;
        this->blockId = GetBlockIdx(); //获取当前blockId
        this->blockNum = GetBlockNum(); //获取当前blockNum

        e_input.SetGlobalBuffer((__gm__ int32_t*)(tempBuffer));
        table_input.SetGlobalBuffer((__gm__ int32_t*)(histogramDevice));
        bits_length_input.SetGlobalBuffer((__gm__ int32_t*)(bits_lengthDevice));
        mbl_output.SetGlobalBuffer((__gm__ uint8_t*)(final + 16 + HISTOGRAM_BINS));
        output.SetGlobalBuffer((__gm__ uint16_t*)(final + 16 + HISTOGRAM_BINS + 32 * blockNum + 2048 * blockNum));

        pipe.InitBuffer(inQueue, BUFFER_NUM, TILE_LEN * sizeof(T));// 初始化Pipe缓冲区，每个Tile大小TILE_LEN
        pipe.InitBuffer(e_outQueue, BUFFER_NUM, DATA_BLOCK_BYTE_NUM / 2 * sizeof(uint8_t));
    }

    __aicore__ inline void Process() {
        pipe.InitBuffer(table, HISTOGRAM_BINS * sizeof(uint32_t));
        LocalTensor<uint32_t> tableLocal = table.Get<uint32_t>();
        DataCopy(tableLocal, table_input, HISTOGRAM_BINS);

        pipe.InitBuffer(bits_length, HISTOGRAM_BINS * sizeof(int32_t));
        LocalTensor<int32_t> blLocal = bits_length.Get<uint32_t>();
        int j = 0;
        int start = 0;
        for(int i = 1; i < HISTOGRAM_BINS; i << 1){
            for(int k = start; k < i; k ++){
                blLocal(k) = j;
            }
            start = i;
            j ++;
        }

        pipe.InitBuffer(calcBuf0, TILE_LEN * sizeof(uint32_t));
        pipe.InitBuffer(calcBuf1, TILE_LEN * sizeof(uint32_t));
        pipe.InitBuffer(calcBuf2, TILE_LEN * sizeof(uint32_t));
        pipe.InitBuffer(calcBuf3, TILE_LEN * sizeof(uint32_t));
        pipe.InitBuffer(calcBuf4, TILE_LEN * sizeof(uint32_t));
        LocalTensor<uint32_t> tempLocal0 = calcBuf0.Get<uint32_t>();
        LocalTensor<uint32_t> tempLocal1 = calcBuf1.Get<uint32_t>();
        LocalTensor<uint32_t> tempLocal2 = calcBuf2.Get<uint32_t>();
        LocalTensor<uint32_t> tempLocal3 = calcBuf3.Get<uint32_t>();
        LocalTensor<uint32_t> tempLocal4 = calcBuf4.Get<uint32_t>();

        pipe.InitBuffer(max_bits_length, TILE_LEN * sizeof(uint32_t));
        LocalTensor<uint32_t> mblLocal = max_bits_length.Get<uint32_t>();

        LocalTensor<uint16_t> e_outLocal = e_outQueue.AllocTensor<uint16_t>();

        for(uint32_t i = blockId; i < blockNum; i += blockNum){
            uint32_t offset0 = i * DATA_BLOCK_BYTE_NUM;
            end = min((int)offset0 + DATA_BLOCK_BYTE_NUM, (int)totalUncompressedBytes);
            this->blockDataBytesSize = end - offset0;
            uint32_t compressedSize = 0;
            uint32_t remainder = 0;
            uint32_t thisTileCompressedSize = 0;
            bool is = 0;
            for(uint32_t tileIdx = 0; tileIdx < TILE_NUM; ++tileIdx){
                uint32_t offset1 = tileIdx * TILE_LEN;
                uint32_t len = min(TILE_LEN, blockDataBytesSize - offset1);
                CopyIn(offset0 + offset1);//输入队列每次都copy32字节
                Compute(is ,tileIdx, compressedSize, remainder, thisTileCompressedSize, mblLocal, e_outLocal[compressedSize], tableLocal, blLocal, byteoffsetLocal, bitsoffsetLocal, tempLocal0, tempLocal1, tempLocal2, tempLocal3);
                CopyOut(is, offset0 + offset1);//当输出块到了32字节就copy到GM
            }
            DataCopy(mbl_output, mblLocal, TILE_NUM);// 每次DataCopy的数据是32字节的倍数
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t offset) {
        LocalTensor<T> inLocal = inQueue.AllocTensor<T>();
        // 拷贝当前Tile数据到Local
        DataCopy(e_inLocal, e_input[offset], len);
        inQueue.EnQue(e_inLocal);
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
    __aicore__ inline void Merge(uint32_t max_bits_length, LocalTensor<int32_t>& encodedData, LocalTensor<uint16_t>& e_outLocal){
        //达到16bit就写出到e_outLocal
        if(max_bits_length == 0){// 最大截断bit = 0，直接不保存
            return;
        }
        uint32_t buffer = 0;
        uint32_t bit_shift = 0;
        uint32_t index = 0;
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

    __aicore__ inline void Compute(bool& is,
                                   uint32_t tileIdx,
                                   uint32_t& compressedSize,
                                   uint32_t& remainder,
                                   uint32_t& thisTileCompressedSize,
                                   LocalTensor<uint8_t>& mblLocal,
                                   LocalTensor<uint16_t>& e_outLocal,
                                   LocalTensor<T>& tableLocal,
                                   LocalTensor<T>& blLocal,
                                   LocalTensor<T>& tempLocal0,
                                   LocalTensor<T>& tempLocal1,
                                   LocalTensor<T>& tempLocal2,
                                   LocalTensor<T>& tempLocal3,
                                   LocalTensor<T>& mergeLocal0,
                                   LocalTensor<T>& mergeLocal1,
                                   LocalTensor<T>& mergeLocal2
    ) {

        LocalTensor<T> e_inLocal = e_inQueue.DeQue<T>();

        And(tempLocal0, e_inLocal, mask2_tensor, len * 2);
        Gather(tempLocal1, table, tempLocal0, (uint32_t)0, len);//gather编码表
        Gather(tempLocal2, blLocal, tempLocal1, (uint32_t)0, len);//gather比特长度
        //求出最大截断bits长度，归约操作
        int32_t max_bits_length0 = 0;
        for(int i = 0; i < TILE_LEN; i ++){
            if(tempLocal2(i) > max_bits_length0){
                max_bits_length0 = tempLocal2(i);
            }
        }
        Merge(max_bits_length0, tempLocal1, e_outLocal);
        // Merge_Vec(max_bits_length0, tempLocal1, e_outLocal, mergeLocal0, mergeLocal1, mergeLocal2);

        ShiftRight(tempLocal0, e_inLocal, (uint32_t)16, len);
        Gather(tempLocal1, table, tempLocal0, (uint32_t)0, len);//gather编码表
        Gather(tempLocal2, blLocal, tempLocal1, (uint32_t)0, len);//gather比特长度
        int32_t max_bits_length1 = 0;
        for(int i = 0; i < TILE_LEN; i ++){
            if(tempLocal2(i) > max_bits_length1){
                max_bits_length1 = tempLocal2(i);
            }
        }
        Merge(max_bits_length1, tempLocal1, e_outLocal);
        // Merge_Vec(max_bits_length1, tempLocal1, e_outLocal[max_bits_length0 << 4], mergeLocal0, mergeLocal1, mergeLocal2);

        thisTileCompressedSize = (max_bits_length0 + max_bits_length1) << 4;
        remainder += thisTileCompressedSize;
        if(remainder > 32){
            remainder -= 32;
            e_outQueue.EnQue(e_outLocal);
            is = true;
        }
        compressedSize += thisTileCompressedSize;
        mblLocal(tile_idx) = (max_bits_length1 << 4) | max_bits_length0;

        inQueue.FreeTensor(e_inLocal);
    }

    __aicore__ inline void CopyOut(bool is, uint32_t offset) {
        if(is){
            LocalTensor<uint16_t> e_outLocal = e_outQueue.DeQue<uint16_t>();//每次输出32字节，应该为16个uint16_t
            DataCopy(output[offset], e_outLocal, TILE_LEN / 2);
        }
        return;
    }

private:
    TPipe* pipe;
    TQue<QuePosition::VECIN, 1> inQueue;
    TQue<QuePosition::VECOUT, 1> e_outQueue;

    TBuf<AscendC::TPosition::VECCALC> table;
    TBuf<AscendC::TPosition::VECCALC> bits_length;
    TBuf<AscendC::TPosition::VECCALC> calcBuf0;
    TBuf<AscendC::TPosition::VECCALC> calcBuf1;
    TBuf<AscendC::TPosition::VECCALC> calcBuf2;
    TBuf<AscendC::TPosition::VECCALC> calcBuf3;
    TBuf<AscendC::TPosition::VECCALC> max_bits_length;
    TBuf<AscendC::TPosition::VECCALC> writeBuf0;
    TBuf<AscendC::TPosition::VECCALC> writeBuf1;
    TBuf<AscendC::TPosition::VECCALC> writeBuf2;

    GlobalTensor<T> e_input;
    GlobalTensor<T> table_input;
    GlobalTensor<int32_t> bits_length_input;
    GlobalTensor<uint8_t> mbl_output;
    GlobalTensor<uint8_t> output;

    uint32_t blockDataBytesSize;
    uint32_t tileNum;
    uint32_t tileLength;
    uint32_t blockId;
    uint32_t blockNum;
    uint32_t compressedSize;//当前压缩后的字节数
};

template<typename T>// T =int32_t
class CoalesceKernel {
public:
    __aicore__ inline PrefixKernel() {}// 计算独占前缀和

    __aicore__ inline void Init(Tpipe& pipe,
                                __gm__ uint8_t* compressedSize,// 输入
                                __gm__ uint8_t* compressedSizePrefix// 输出
                                
    ) {
        this->pipe = pipe;
        input.SetGlobalBuffer((__gm__ T*)(compressedSize));
        output.SetGlobalBuffer((__gm__ T*)(compressedSizePrefix));

        pipe.InitBuffer(inQueue, BUFFER_NUM, DATA_BLOCK_NUM * sizeof(T));
        pipe.InitBuffer(outQueue, BUFFER_NUM, DATA_BLOCK_NUM * sizeof(T));
    }

    __aicore__ inline void Process() {
        CopyIn();
        Compute();
        CopyOut();
    }

private:
    __aicore__ inline void CopyIn() {
        LocalTensor<T> inLocal = inQueue.AllocTensor<T>();
        DataCopy(inLocal, input, DATA_BLOCK_NUM);
        inQueue.EnQue(inLocal);
    }

    __aicore__ inline void Compute() {
        LocalTensor<T> inLocal = inQueue.DeQue<T>();
        LocalTensor<T> outLocal = outQueue.AllocTensor<T>();
        for(int l = 1; l < DATA_BLOCK_NUM; l ++){
            outLocal(l) = outLocal(l - 1);
        }
        for(int l = 0; l < log2(DATA_BLOCK_NUM); l ++){
            for (int i = (1 << l); i < n; i++)
                outLocal(i) += outLocal(i - (1 << l));
        }
        inQueue.FreeTensor(inLocal);
        outQueue.EnQue(outLocal);
    }

    __aicore__ inline void CopyOut() {
        LocalTensor<T> outLocal = outQueue.DeQue<T>();
        DataCopy(output, outLocal, DATA_BLOCK_NUM);
        outQueue.FreeTensor(outLocal);
    }

private:
    TPipe* pipe;
    TQue<QuePosition::VECIN, 1> inQueue;
    TQue<QuePosition::VECOUT, 1> outQueue;

    GlobalTensor<T> input;
    GlobalTensor<T> output;
};

template<typename T>
class CoalesceKernel {
public:
    __aicore__ inline CoalesceKernel() {} // 生成数据头，紧缩码字
    // 输入：max_bit_length数组（3bits * block_num），码字（max_bit_length * blockSize）
    // 输出：一整块连续的压缩块，压缩块的大小

    __aicore__ inline void Init(Tpipe& pipe,
                                uint32_t dataBlockNum,
                                __gm__ uint8_t* tempBuffer1, //e_input
                                __gm__ uint8_t* finalCompressedExp, //output
                                __gm__ uint8_t* compressedSize,
                                __gm__ uint8_t* compressedSizePrefix,
                                uint32_t totalUncompressedBytes) {
        this->pipe = pipe;
        this->dataBlockNum = dataBlockNum;
        this->blockId = GetBlockIdx();
        this->blockNum = GetBlockNum();

        input.SetGlobalBuffer((__gm__ T*)(tempBuffer1));
        output.SetGlobalBuffeT((__gm__ T*)(finalCompressedExp));

        pipe.InitBuffer(queBind, BUFFER_NUM, TILE_LEN * sizeof(T));
    }

private:

    __aicore__ inline void Process()
    {
        auto bindLocal = queBind.AllocTensor<T>();
        for(int i = blockId; i < dataBlockNum; i += blockNum){
            int compSize = compressedSize[i];
            int compSizePrefix = compressedSizePrefix[i];//字节为单位
            int count = compSize / 32;//每次运输32字节的数据
            for(int j = 0; j < count; j ++){
                DataCopy(bindLocal, input[i * 4096 + j * TILE_LEN], TILE_LEN);
                queBind.EnQue(bindLocal);
                queBind.DeQue(bindLocal);
                DataCopy(output[compSizePrefix + j * TILE_LEN], bindLocal, TILE_LEN);
            }
        }
        queBind.FreeTensor(bindLocal);
    }

private:
    TPipe* pipe;
    TQueBind<TPosition::VECIN, TPosition::VECOUT, BUFFER_NUM> queBind;

    GlobalTensor<T> input;//输入每个数据块压缩后的GM地址
    GlobalTensor<T> output;//输出每个数据块压缩后的GM地址
    GlobalTensor<T> compressedSize;
    GlobalTensor<T> compressedSizePrefix;

    uint32_t blockId;
    uint32_t blockNum;
    uint32_t dataBlockNum;
};

__global__ __aicore__ void extractbits_and_histogram(...)
{
    TPipe pipe;
    Extractbits_and_histogramKernel<float> op;
    op.Init(..., &pipe);
    op.process();
}

__global__ __aicore__ void MergeHistogramKernel(...)
{
    TPipe pipe;
    MergeHistogramKernel<int32_t> op;
    op.Init(..., &pipe);
    op.process();
}

void* generate_table(int32_t* histogramDevice, uint32_t* table){
    uint64_t sortDevice[HISTOGRAM_BINS];
    for(int i = 0; i < HISTOGRAM_BINS; i ++){
        sortDevice[i] = ((uint64_t)histogramDevice[i] << 32) | i;
    }
    std::sort(sortDevice, sortDevice + HISTOGRAM_BINS, [](uint64_t a, uint64_t b) { return a > b;});
    for(int i = 0; i < HISTOGRAM_BINS; i ++){
        histogramDevice[(int)(sortDevice[i] & 0x0000000011111111)] = i;
    }
    for(int i = 0; i < HISTOGRAM_BINS; i ++){
        table[i] = (uint8_t)histogramDevice[i];
    }
}

__global__ __aicore__ void comp(...)
{
    TPipe pipe;
    KernelExample<float> op;
    op.Init(..., &pipe);
    op.process();
}

__global__ __aicore__ void calcprefix(...)
{
    TPipe pipe;
    KernelExample<float> op;
    op.Init(..., &pipe);
    op.process();
}

__global__ __aicore__ void coalesce(...)
{
    TPipe pipe;
    KernelExample<float> op;
    op.Init(..., &pipe);
    op.process();
}

extern "C" void compress(uint32_t blockNum, nullptr, void* stream, uint8_t* srcDevice, uint8_t* tempBuffer, uint8_t* final, int32_t* histogramDevice, uint32_t* compressedSize, uint32_t* compressedSizePrefix, uint32_t totalCompressedSize) {
    extractbits_and_histogram<<<blockNum, nullptr, stream>>>(srcDevice, tempBuffer, final);//提取字节并计算直方图
    MergeHistogramKernel<<<blockNum, nullptr, stream>>>();
    generate_table(histogramDevice, final + 16);//排序后table(uint8_t数组)直接写进final区域，uint32_t的histogram用于压缩
    comp<<<blockNum, nullptr, stream>>>(tempBuffer, final, reinterpret_cast<uint8_t*>(histogramDevice), reinterpret_cast<uint8_t*>(compressedSize));//压缩函数
    calcprefix<<<1, nullptr, stream>>>(reinterpret_cast<uint8_t*>(compressedSize), reinterpret_cast<uint8_t*>(compressedSizePrefix));//计算前缀和，用于后续块合并，字节为单位，
    coalesce<<<blockNum, nullptr, stream>>>(tempBuffer, final, reinterpret_cast<uint8_t*>(compressedSize), reinterpret_cast<uint8_t*>(compressedSizePrefix));//纯搬运内核
}

// // 注册算子
// __attribute__((visibility("default"))) 
// void RegisterExtractBitsKernel() {
//     KernelRegistrar<extractbits>()
//         .Input(GM_TYPE_UINT16)
//         .Output(GM_TYPE_UINT8, "e_output")
//         .Output(GM_TYPE_UINT8, "m_s_output")
//         .Attr("totalElements", REQUIRED_ATTR);
// }


