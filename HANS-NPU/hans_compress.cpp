#include "kernel_operator.h"
#include "hans_utils.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2; 
constexpr int32_t BLOCK_NUM = 256;
constexpr uint32_t HISTOGRAM_BINS = 256;
constexpr uint32_t HANDLE_NUM_PER = 32; // 算子每次向量化处理32单位的数据量，直方图计算中每个block生成32个temp_table，一共32 * 4 * 256 = 32KB
constexpr uint32_t TILE_LEN = 32; // 每个Tile处理32个单元(单元指输入数据的类型)

template<typename T>
class ExtractBits1Kernel {
public:
    __aicore__ inline ExtractBits1Kernel() {} // 切分数据，分离指数位，同时进行histogram统计
    // 输入：uint16_t数组(两两组成一个int32_t)
    // 输出：指数数组（int32_t），尾数+sign数组（uint8_t)，histogram统计数组（int32_t）

    __aicore__ inline void Init(__gm__ uint8_t* in, 
                                __gm__ uint8_t* e_out0, 
                                __gm__ uint8_t* e_out1,
                                __gm__ uint8_t* m_s_out, 
                                __gm__ uint8_t* hist_out, 
                                uint32_t inblockElements) {
        input.SetGlobalBuffer((__gm__ uint32_t*)in);
        e_output0.SetGlobalBuffer((__gm__ uint32_t*)e_out0);
        e_output1.SetGlobalBuffer((__gm__ uint32_t*)e_out1);
        m_s_output.SetGlobalBuffer((__gm__ uint32_t*)m_s_out);
        hist_output.SetGlobalBuffer((__gm__ int32_t*)hist_out);

        this->blockElements = inblockElements ;
        // / sizeof(T);//假设block处理的数据量为32字节的倍数,inblockElements以字节为单位，blockElements以sizeof(T)字节为单位
        this->tileNum = (blockElements + TILE_LEN - 1) / (TILE_LEN); //每次处理32个int32_t的数据量，计算处理的次数

        // assert(tileNum == 4);
        // 初始化Pipe缓冲区，每个Tile大小TILE_LEN
        pipe.InitBuffer(inQueue, BUFFER_NUM, TILE_LEN * sizeof(T));
        pipe.InitBuffer(e_outQueue0, BUFFER_NUM, TILE_LEN * sizeof(uint32_t));
        pipe.InitBuffer(e_outQueue1, BUFFER_NUM, TILE_LEN * sizeof(uint32_t));
        pipe.InitBuffer(m_s_outQueue, BUFFER_NUM, TILE_LEN * sizeof(uint32_t));
        // pipe.InitBuffer(hist_outQueue, BUFFER_NUM, HISTOGRAM_BINS * sizeof(int32_t));
        //因为开启了double_buffer，最多只能开四个queue
    }

    __aicore__ inline void Process() {
        pipe.InitBuffer(mask0, TILE_LEN * sizeof(uint32_t));
        LocalTensor<T> mask0_tensor = mask0.Get<T>();
        Duplicate(mask0_tensor, (uint32_t)65280, TILE_LEN);//11111111 00000000

        pipe.InitBuffer(mask1, TILE_LEN * sizeof(uint32_t));
        LocalTensor<T> mask1_tensor = mask1.Get<T>();
        Duplicate(mask1_tensor, (uint32_t)255, TILE_LEN);//00000000 11111111

        pipe.InitBuffer(mask2, TILE_LEN * sizeof(uint32_t));
        LocalTensor<T> mask2_tensor = mask2.Get<T>();
        Duplicate(mask2_tensor, (uint32_t)16711935, TILE_LEN);
        //00000000 11111111 00000000 11111111

        pipe.InitBuffer(mask3, TILE_LEN * sizeof(uint32_t));
        LocalTensor<T> mask3_tensor = mask3.Get<T>();
        Duplicate(mask3_tensor, (uint32_t)4278255360, TILE_LEN);
        //11111111 00000000 11111111 00000000

        pipe.InitBuffer(mask4, TILE_LEN * sizeof(uint32_t));
        LocalTensor<T> mask4_tensor = mask4.Get<T>();
        Duplicate(mask4_tensor, (uint32_t)65535, TILE_LEN);
        //00000000 00000000 11111111 11111111

        pipe.InitBuffer(one, TILE_LEN * sizeof(int32_t));
        LocalTensor<int32_t> all_one = one.Get<int32_t>();
        Duplicate(all_one, (int32_t)1, TILE_LEN);//0000 0000 0000 0001

        pipe.InitBuffer(offset, TILE_LEN * sizeof(int32_t));
        LocalTensor<int32_t> offset_tensor = offset.Get<int32_t>();
        uint32_t num = ((1 << 16) + 1) << 10;
        for(int i = 0; i < TILE_LEN; i ++){
            offset_tensor(i) = i * num;
        }

        pipe.InitBuffer(tempHist, TILE_LEN * 
        HISTOGRAM_BINS * sizeof(int32_t));
        LocalTensor<int32_t> histogram = tempHist.Get<int32_t>();
        Duplicate(histogram, (int32_t)0, HISTOGRAM_BINS);

        pipe.InitBuffer(calcBuf0, TILE_LEN * sizeof(uint32_t));
        pipe.InitBuffer(calcBuf1, TILE_LEN * sizeof(uint32_t));
        pipe.InitBuffer(calcBuf2, TILE_LEN * sizeof(uint32_t));
        pipe.InitBuffer(calcBuf3, TILE_LEN * sizeof(uint32_t));
        LocalTensor<uint32_t> tempLocal0 = calcBuf0.Get<uint32_t>();
        LocalTensor<uint32_t> tempLocal1 = calcBuf1.Get<uint32_t>();
        LocalTensor<uint32_t> tempLocal2 = calcBuf2.Get<uint32_t>();
        LocalTensor<uint32_t> tempLocal3 = calcBuf3.Get<uint32_t>();

        pipe.InitBuffer(histBuffer0, TILE_LEN * sizeof(int32_t));
        pipe.InitBuffer(histBuffer1, TILE_LEN * sizeof(int32_t));
        LocalTensor<int32_t> histTensor0 = histBuffer0.Get<int32_t>();
        LocalTensor<int32_t> histTensor1 = histBuffer1.Get<int32_t>();

        for (uint32_t tileIdx = 0; tileIdx < //1
        tileNum
        ; ++tileIdx) {
            uint32_t offset = tileIdx * TILE_LEN;
            uint32_t len = min((int)TILE_LEN, (int)(blockElements - offset));
            // uint32_t len = TILE_LEN;
            CopyIn(tileIdx, offset, len);
            Compute(tileIdx, len, mask0_tensor, mask1_tensor, mask2_tensor, mask3_tensor, mask4_tensor, all_one, offset_tensor, histogram, tempLocal0, tempLocal1, tempLocal2, tempLocal3, histTensor0, histTensor1);
            CopyOut(tileIdx, offset, len);
        }
        // assert(histogram(0) == 111);
        // LocalTensor<int32_t> histLocal = hist_outQueue.AllocTensor<int32_t>();
        // for(int i = 0; i < HISTOGRAM_BINS; i ++)
        // {
        //     histLocal(i) = 111;
        // }
        // DataCopy(hist_output, histLocal, HISTOGRAM_BINS);
        MergeHistogram(histogram);// 合并32个temp直方图为最终的一个
    }

private:
    __aicore__ inline void CopyIn(uint32_t tileIdx, uint32_t offset, uint32_t len) {
        LocalTensor<T> inLocal = inQueue.AllocTensor<T>();
        
        // 拷贝当前Tile数据到Local
        DataCopy(inLocal, input[offset], len);
        inQueue.EnQue(inLocal);
    }

    __aicore__ inline void Compute(uint32_t tileIdx, 
                                   uint32_t len,
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
        LocalTensor<uint32_t> e_outLocal0 = e_outQueue0.AllocTensor<uint32_t>();
        LocalTensor<uint32_t> e_outLocal1 = e_outQueue1.AllocTensor<uint32_t>();
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
        // assert(tempLocal0.GetValue(0) == 4294967294);
        // assert(tempLocal0.GetValue(16) == 4294967294
        // // 0xFFFFFFFF
        // );
        // assert(tempLocal1.GetValue(0) == 1);
        // assert(tempLocal1.GetValue(16) == 1);
        // assert(tempLocal2.GetValue(0) == 4294967295);
        // assert(tempLocal2.GetValue(16) == 4294967295);

        And(m_s_outLocal, tempLocal2, mask2_tensor, (int32_t)len * 2);//取出从高到低1和3字节
        And(tempLocal3, tempLocal2, mask3_tensor, (int32_t)len * 2);//取出从高到低0和2字节
        ShiftRight(tempLocal1, tempLocal3, (uint32_t)8, (int32_t)len);//右移8位
        And(e_outLocal0, tempLocal1, mask1_tensor, (int32_t)len * 2);//取出低8位
        ShiftRight(tempLocal0, tempLocal1, (uint32_t)16, (int32_t)len);//右移16位
        And(e_outLocal1, tempLocal0, mask1_tensor, (int32_t)len * 2);//取出低8位
        ShiftRight(tempLocal1, tempLocal3, (uint32_t)6, (int32_t)len);
        Add(tempLocal2.template ReinterpretCast<int32_t>(), tempLocal1.template ReinterpretCast<int32_t>(), offset_tensor, (int32_t)len);
        // assert(tempLocal1.GetValue(0) == 4653194);
        // assert(tempLocal2.GetValue(0) == 4653194);
        //同时对两个位置加offset

        // And(e_outLocal0, tempLocal1, mask1_tensor, (int32_t)len * 2);//取出低8位
        // // assert(e_outLocal0.GetValue(0) == 128);
        // ShiftRight(tempLocal0, tempLocal1, (uint32_t)16, (int32_t)len);//右移16位
        // And(e_outLocal1, tempLocal0, mask1_tensor, (int32_t)len * 2);//取出低8位

        And(//e_outLocal0,
            tempLocal0, 
            tempLocal2, mask4_tensor, (int32_t)len * 2);//取出低16位
        // assert(tempLocal0(0) == 1020);
        Gather(histTensor0, histogram, tempLocal0, (uint32_t)0, (uint32_t)len);
        Add(//histogram,
            histTensor1, 
            histTensor0, all_one, (int32_t)len);
        // assert(histTensor1(0) == 1);
        for(int i = 0; i < len; i ++){
            histogram(tempLocal0(i) >> 2) = histTensor1(i);
        }
        // Scatter(histogram.template ReinterpretCast<uint32_t>(), histTensor1.template ReinterpretCast<uint32_t>(), tempLocal0, (uint32_t)0, (uint32_t)len);
        // assert(histogram(255) == 1);

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

        // And(m_s_outLocal, tempLocal2, mask2_tensor, len);

        // ShiftRight(tempLocal2, tempLocal2, (uint32_t)8, len);
        // // And(e_outLocal0, tempLocal2, mask2_tensor, len);
        // // Add(e_outLocal1, e_outLocal0, offsetLocal, len);
        // And(tempLocal1, tempLocal2, mask1_tensor, len);

        // ShiftRight(tempLocal0, tempLocal2, (uint32_t)8, len);
        // And(e_outLocal0, tempLocal0, mask1_tensor, len);
        // // assert(e_outLocal0.GetValue(0) == 255);

        // And(tempLocal2, tempLocal0, mask0_tensor, len);
        // Or(m_s_outLocal, tempLocal1, tempLocal2, len);

        // ShiftRight(tempLocal2, tempLocal0, (uint32_t)16, len);
        // And(e_outLocal1, tempLocal2, mask1_tensor, len);


        // Add(tempLocal0.template ReinterpretCast<int32_t>(), e_outLocal0.template ReinterpretCast<int32_t>(), offsetLocal.template ReinterpretCast<int32_t>(), (int32_t)len);
        // Gather(tempLocal1, histogram, tempLocal0, (uint32_t)0, (uint32_t)len);
        // Add(tempLocal2.template ReinterpretCast<int32_t>(), tempLocal1.template ReinterpretCast<int32_t>(), all_one.template ReinterpretCast<int32_t>(), (int32_t)len);
        // Scatter(histogram, tempLocal2, tempLocal0, (uint32_t)0, (uint32_t)len);

        // Add(tempLocal0.template ReinterpretCast<int32_t>(), e_outLocal1.template ReinterpretCast<int32_t>(),  offsetLocal.template ReinterpretCast<int32_t>(), (int32_t)len);
        // Gather(tempLocal1, histogram, tempLocal0, (uint32_t)0, (uint32_t)len);
        // Add(tempLocal2.template ReinterpretCast<int32_t>(), tempLocal1.template ReinterpretCast<int32_t>(), all_one.template ReinterpretCast<int32_t>(), (int32_t)len);
        // Scatter(histogram, tempLocal2, tempLocal0, (uint32_t)0, (uint32_t)len);

        inQueue.FreeTensor(inLocal);
        e_outQueue0.EnQue(e_outLocal0);
        e_outQueue1.EnQue(e_outLocal1);
        m_s_outQueue.EnQue(m_s_outLocal);
    }

    __aicore__ inline void CopyOut(uint32_t tileIdx, uint32_t offset, uint32_t len) {
        LocalTensor<uint32_t> e_outLocal0 = e_outQueue0.DeQue<uint32_t>();
        LocalTensor<uint32_t> e_outLocal1 = e_outQueue1.DeQue<uint32_t>();
        LocalTensor<uint32_t> m_s_outLocal = m_s_outQueue.DeQue<uint32_t>();

        // 将结果拷贝回Global Memory
        DataCopy(e_output0[offset], e_outLocal0, len);
        DataCopy(e_output1[offset], e_outLocal1, len);
        DataCopy(m_s_output[offset], m_s_outLocal, len);

        e_outQueue0.FreeTensor(e_outLocal0);
        e_outQueue1.FreeTensor(e_outLocal1);
        m_s_outQueue.FreeTensor(m_s_outLocal);
    }

    __aicore__ inline void MergeHistogram(LocalTensor<int32_t>& histogram) {
        
        int sum = 0;
        for (uint32_t i = 1; i < TILE_LEN; ++i) {
            for (uint32_t bin = 0; bin < HISTOGRAM_BINS; ++bin) {
                histogram(bin) += histogram(i * HISTOGRAM_BINS + bin);
            }
        }
        for(int i = 0; i < HISTOGRAM_BINS; i ++){
            sum += histogram(i);
        }
        assert(sum == 256);
        // LocalTensor<int32_t> hist_outLocal = m_s_outQueue.AllocTensor<int32_t>();
        // for(int i = 0; i < 256; i ++)
        //     hist_outLocal(i) = histogram(i);
        DataCopy(hist_output, histogram, HISTOGRAM_BINS);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> inQueue;
    TQue<QuePosition::VECOUT, 1> e_outQueue0;
    TQue<QuePosition::VECOUT, 1> e_outQueue1;
    TQue<QuePosition::VECOUT, 1> m_s_outQueue;
    // TQue<QuePosition::VECOUT, 1> hist_outQueue;
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
    TBuf<TPosition::VECCALC> offset;
    TBuf<TPosition::VECCALC> one;
    GlobalTensor<T> input;
    GlobalTensor<uint32_t> e_output0;
    GlobalTensor<uint32_t> e_output1;
    GlobalTensor<uint32_t> m_s_output;
    GlobalTensor<int32_t> hist_output;
    uint32_t blockElements;//以sizeof(T)为单位的数据量
    uint32_t tileNum;
};

extern "C" __global__ __aicore__ void kernel_ExtractBits1(GM_ADDR inGm, GM_ADDR eGm0, GM_ADDR eGm1, GM_ADDR msGm, GM_ADDR hist, uint32_t length)
{
    ExtractBits1Kernel<uint32_t> op; 
    op.Init(inGm, eGm0, eGm1, msGm, hist, length);
    op.Process();
}

void ExtractBits1_do(uint32_t blockDim, void *stream, uint8_t *inGm, uint8_t *eGm0, uint8_t *eGm1, uint8_t *msGm, uint8_t* hist, uint32_t length)
{
    kernel_ExtractBits1<<<blockDim, nullptr, stream>>>(inGm, eGm0, eGm1, msGm, hist, length);
}
// extern "C" 
__global__ __aicore__ void extractbits(GM_ADDR input, GM_ADDR e_output, 
                                                 GM_ADDR m_s_output, GM_ADDR totalElements) {
    // 获取总元素数
    uint32_t total = *(reinterpret_cast<const uint32_t*>(totalElements));
    uint32_t blockId = GetBlockIdx();
    uint32_t blockNum = GetBlockNum();

    // 计算当前Block处理的数据范围
    uint32_t perBlock = (total + blockNum - 1) / blockNum;
    uint32_t start = blockId * perBlock;
    if (start >= total) return;
    uint32_t end = min(start + perBlock, total);
    uint32_t blockElements = end - start;

    // 调整输入输出指针
    GlobalTensor<uint16_t> inputGm(input + start * sizeof(uint16_t));
    GlobalTensor<uint8_t> eOutputGm(e_output + start * sizeof(uint8_t));
    GlobalTensor<uint8_t> msOutputGm(m_s_output + start * sizeof(uint8_t));

    // 初始化并处理
    ExtractBitsKernel<uint16_t> op;
    op.Init(inputGm, eOutputGm, msOutputGm, blockElements);
    op.Process();
}

template<typename T>
class TableKernel { // 进行histogram合并与sym排序，生成编码表
public:
    __aicore__ inline TableKernel() {} // 切分数据，分离指数位
    // 输入：多block的histogram数组
    // 输出：排序好的table表（uint8_t）

    __aicore__ inline void Init(GlobalTensor<T>& input, GlobalTensor<uint8_t>& output, 
                               uint32_t blockElements) {
        this->input = input;
        this->e_output = e_output;
        this->m_s_output = m_s_output;
        this->blockElements = blockElements;
        this->tileNum = tileNum;
        this->tileLength = blockElements / tileNum / BUFFER_NUM;
        //(totalElements + TILE_LEN - 1) / TILE_LEN;

        // 初始化Pipe缓冲区，每个Tile大小TILE_LEN
        pipe.InitBuffer(inQueue, BUFFER_NUM, TILE_LEN * sizeof(T));
        pipe.InitBuffer(e_outQueue, BUFFER_NUM, TILE_LEN * sizeof(uint8_t));
        pipe.InitBuffer(m_s_outQueue, BUFFER_NUM, TILE_LEN * sizeof(uint8_t));
    }

    __aicore__ inline void Process() {
        int32_t loopCount = this->tileNum * BUFFER_NUM;
        for (uint32_t tileIdx = 0; tileIdx < loopCount; ++tileIdx) {
            CopyIn(tileIdx);
            Compute(tileIdx);
            CopyOut(tileIdx);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t tileIdx) {
        uint32_t offset = tileIdx * TILE_LEN;
        uint32_t copyLen = min(TILE_LEN, totalElements - offset);
        LocalTensor<T> inLocal = inQueue.AllocTensor<T>();
        
        // 拷贝当前Tile数据到Local
        DataCopy(inLocal, input[offset], copyLen);
        inQueue.EnQue(inLocal);
    }

    __aicore__ inline void Compute(uint32_t tileIdx) {
        uint32_t offset = tileIdx * TILE_LEN;
        uint32_t computeLen = min(TILE_LEN, totalElements - offset);
        LocalTensor<T> inLocal = inQueue.DeQue<T>();
        LocalTensor<uint8_t> e_outLocal = e_outQueue.AllocTensor<uint8_t>();
        LocalTensor<uint8_t> m_s_outLocal = m_s_outQueue.AllocTensor<uint8_t>();

        // 处理每个元素
        for (uint32_t i = 0; i < computeLen; ++i) {
            uint16_t val = inLocal.GetValue(i);
            uint32_t extracted_temp = (val << 16) | val; // 两个相同值直接与
            uint8_t extracted_e = (extracted_temp >> 7) & 0xFF;  // 提取高8位
            uint8_t extracted_m_s = (extracted_temp >> 15) & 0xFF;
            e_outLocal.SetValue(i, e);
            m_s_outLocal.SetValue(i, m_s);
        }

        inQueue.FreeTensor(inLocal);
        e_outQueue.EnQue(e_outLocal);
        m_s_outQueue.EnQue(m_s_outLocal);
    }

    __aicore__ inline void CopyOut(uint32_t tileIdx) {
        uint32_t offset = tileIdx * TILE_LEN;
        uint32_t copyLen = min(TILE_LEN, totalElements - offset);
        LocalTensor<uint8_t> e_outLocal = e_outQueue.DeQue<uint8_t>();
        LocalTensor<uint8_t> m_s_outLocal = m_s_outQueue.DeQue<uint8_t>();

        // 将结果拷贝回Global Memory
        DataCopy(e_output[offset], e_outLocal, copyLen);
        DataCopy(m_s_output[offset], m_s_outLocal, copyLen);

        e_outQueue.FreeTensor(e_outLocal);
        m_s_outQueue.FreeTensor(m_s_outLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> e_outQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> m_s_outQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> hist_outQueue;
    TBuf<AscendC::TPosition::VECCALC> calcBuf0;
    TBuf<AscendC::TPosition::VECCALC> calcBuf1;
    TBuf<AscendC::TPosition::VECCALC> calcBuf2;
    GlobalTensor<T> input;
    GlobalTensor<uint32_t> e_output;
    GlobalTensor<uint8_t> m_s_output;
    uint32_t blockElements;
    uint32_t tileNum;
};

template<typename T>
class CompressKernel {
public:
    __aicore__ inline CompressKernel() {}
    // 输入：指数数组（uint8_t），table(uint8_t)，
    // 输出：max_bit_length数组（3bits * block_num），码字（max_bit_length * blockSize）

    __aicore__ inline void Init(GlobalTensor<T>& e_input, GlobalTensor<uint8_t>& table_input, 
                               GlobalTensor<uint8_t>& m_s_output, uint32_t blockElements, uint32_t tileNum) {
        this->input = input;
        this->e_output = e_output;
        this->m_s_output = m_s_output;
        this->blockElements = blockElements;
        this->tileNum = tileNum;
        this->tileLength = blockElements / tileNum / BUFFER_NUM;
        //(totalElements + TILE_LEN - 1) / TILE_LEN;

        // 初始化Pipe缓冲区，每个Tile大小TILE_LEN
        pipe.InitBuffer(inQueue, BUFFER_NUM, TILE_LEN * sizeof(T));
        pipe.InitBuffer(e_outQueue, BUFFER_NUM, TILE_LEN * sizeof(uint8_t));
        pipe.InitBuffer(m_s_outQueue, BUFFER_NUM, TILE_LEN * sizeof(uint8_t));
    }

    __aicore__ inline void Process() {
        int32_t loopCount = this->tileNum * BUFFER_NUM;
        for (uint32_t tileIdx = 0; tileIdx < loopCount; ++tileIdx) {
            CopyIn(tileIdx);
            Compute(tileIdx);
            CopyOut(tileIdx);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t tileIdx) {
        uint32_t offset = tileIdx * TILE_LEN;
        uint32_t copyLen = min(TILE_LEN, totalElements - offset);
        LocalTensor<T> inLocal = inQueue.AllocTensor<T>();
        
        // 拷贝当前Tile数据到Local
        DataCopy(inLocal, input[offset], copyLen);
        inQueue.EnQue(inLocal);
    }

    __aicore__ inline void Compute(uint32_t tileIdx) {
        uint32_t offset = tileIdx * TILE_LEN;
        uint32_t computeLen = min(TILE_LEN, totalElements - offset);
        LocalTensor<T> e_inLocal = e_inQueue.DeQue<T>();
        LocalTensor<uint8_t> ecd_outLocal = e_outQueue.AllocTensor<uint8_t>();
        LocalTensor<uint8_t> mbl_outLocal = mbl_outQueue.AllocTensor<uint8_t>();
        LocalTensor<T> tempLocal0 = m_queOut.AllocTensor<T>();
        LocalTensor<T> tempLocal1 = ?.AllocTensor<T>();
        LocalTensor<T> tempLocal2 = ?.AllocTensor<T>();

        // Gather(dstLocal, srcLocal, srcOffsetLocal, (uint32_t)0, m_elementCount);
        Gather(tempLocal0, tableLocal, inLocal, (uint32_t)0, TILE_LEN); 
        Gather(tempLocal1, clzLocal, tempLocal0, (uint32_t)0, TILE_LEN);
        ReduceMax(const LocalTensor<T>& tempLocal2, const LocalTensor<T>& tempLocal1, const LocalTensor<T>& workLocal, TILE_LEN)
        uint8_t max_bits_length = *tempLocal2;
        int buffer_num = divUp(8, max_bits_length);
        // LocalTensor<T> buffer[buffer_num][TILE_LEN];

        LocalTensor<T> offset;
        Add(buffer_offset, tempLocal0, offset, TILE_LEN);
        LocalTensor<T> temp;
        gather(temp, buffer, buffer_offset, (uint32_t)0, TILE_LEN);
        Or(temp, );
        // LocalTensor<T> tempbuffer[2];
        // LocalTensor<T> tempbuffer;
        for(int i = 0; i < buffer_num; i++){
            Or(buffer[0], buffer[0], buffer[i], TILE_LEN);
        }

        // // 处理每个元素
        // for (uint32_t i = 0; i < computeLen; ++i) {
        //     uint16_t val = inLocal.GetValue(i);
        //     uint32_t extracted_temp = (val << 16) | val; // 两个相同值直接与
        //     uint8_t extracted_e = (extracted_temp >> 7) & 0xFF;  // 提取高8位
        //     uint8_t extracted_m_s = (extracted_temp >> 15) & 0xFF;
        //     ecd_outLocal.SetValue(i, e);
        //     mbl_outLocal.SetValue(i, m_s);
        // }

        inQueue.FreeTensor(e_inLocal);
        ecd_outQueue.EnQue(ecd_outLocal);
        mbl_outQueue.EnQue(mbl_outLocal);
    }

    __aicore__ inline void CopyOut(uint32_t tileIdx) {
        uint32_t offset = tileIdx * TILE_LEN;
        uint32_t copyLen = min(TILE_LEN, totalElements - offset);
        LocalTensor<uint8_t> ecd_outLocal = ecd_outQueue.DeQue<uint8_t>();
        LocalTensor<uint8_t> mbl_outLocal = m_s_outQueue.DeQue<uint8_t>();

        // 将结果拷贝回Global Memory
        DataCopy(ecd_output[offset], ecd_outLocal, copyLen);
        DataCopy(mbl_output[offset], mbl_outLocal, copyLen);

        ecd_outQueue.FreeTensor(e_outLocal);
        mbl_outQueue.FreeTensor(m_s_outLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> e_inQueue;
    TQue<QuePosition::VECIN, BUFFER_NUM> table_inQueue;
    TQue<QuePosition::VECIN, BUFFER_NUM> clz_inQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> ecd_outQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> mbl_outQueue;
    TBuf<AscendC::TPosition::VECCALC> histogram;
    GlobalTensor<T> e_input, table_input, clz_input;
    GlobalTensor<uint8_t> ecd_output, mbl_output;
    uint32_t blockElements;
    uint32_t tileNum;
    uint32_t tileLength;
};


template<typename T>
class CoalesceKernel {
public:
    __aicore__ inline CoalesceKernel() {} // 生成数据头，紧缩码字
    // 输入：max_bit_length数组（3bits * block_num），码字（max_bit_length * blockSize）
    // 输出：一整块连续的压缩块，压缩块的大小

    __aicore__ inline void Init(GlobalTensor<T>& input, GlobalTensor<uint8_t>& e_output, 
                               GlobalTensor<uint8_t>& m_s_output, uint32_t blockElements, uint32_t tileNum) {
        this->input = input;
        this->e_output = e_output;
        this->m_s_output = m_s_output;
        this->blockElements = blockElements;
        this->tileNum = tileNum;
        this->tileLength = blockElements / tileNum / BUFFER_NUM;
        //(totalElements + TILE_LEN - 1) / TILE_LEN;

        // 初始化Pipe缓冲区，每个Tile大小TILE_LEN
        pipe.InitBuffer(inQueue, BUFFER_NUM, TILE_LEN * sizeof(T));
        pipe.InitBuffer(e_outQueue, BUFFER_NUM, TILE_LEN * sizeof(uint8_t));
        pipe.InitBuffer(m_s_outQueue, BUFFER_NUM, TILE_LEN * sizeof(uint8_t));
    }

    __aicore__ inline void Process() {
        int32_t loopCount = this->tileNum * BUFFER_NUM;
        for (uint32_t tileIdx = 0; tileIdx < loopCount; ++tileIdx) {
            CopyIn(tileIdx);
            Compute(tileIdx);
            CopyOut(tileIdx);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t tileIdx) {
        uint32_t offset = tileIdx * TILE_LEN;
        uint32_t copyLen = min(TILE_LEN, totalElements - offset);
        LocalTensor<T> inLocal = inQueue.AllocTensor<T>();
        
        // 拷贝当前Tile数据到Local
        DataCopy(inLocal, input[offset], copyLen);
        inQueue.EnQue(inLocal);
    }

    __aicore__ inline void Compute(uint32_t tileIdx) {
        uint32_t offset = tileIdx * TILE_LEN;
        uint32_t computeLen = min(TILE_LEN, totalElements - offset);
        LocalTensor<T> inLocal = inQueue.DeQue<T>();
        LocalTensor<uint8_t> e_outLocal = e_outQueue.AllocTensor<uint8_t>();
        LocalTensor<uint8_t> m_s_outLocal = m_s_outQueue.AllocTensor<uint8_t>();

        // 处理每个元素
        for (uint32_t i = 0; i < computeLen; ++i) {
            uint16_t val = inLocal.GetValue(i);
            uint32_t extracted_temp = (val << 16) | val; // 两个相同值直接与
            uint8_t extracted_e = (extracted_temp >> 7) & 0xFF;  // 提取高8位
            uint8_t extracted_m_s = (extracted_temp >> 15) & 0xFF;
            e_outLocal.SetValue(i, e);
            m_s_outLocal.SetValue(i, m_s);
        }

        inQueue.FreeTensor(inLocal);
        e_outQueue.EnQue(e_outLocal);
        m_s_outQueue.EnQue(m_s_outLocal);
    }

    __aicore__ inline void CopyOut(uint32_t tileIdx) {
        uint32_t offset = tileIdx * TILE_LEN;
        uint32_t copyLen = min(TILE_LEN, totalElements - offset);
        LocalTensor<uint8_t> e_outLocal = e_outQueue.DeQue<uint8_t>();
        LocalTensor<uint8_t> m_s_outLocal = m_s_outQueue.DeQue<uint8_t>();

        // 将结果拷贝回Global Memory
        DataCopy(e_output[offset], e_outLocal, copyLen);
        DataCopy(m_s_output[offset], m_s_outLocal, copyLen);

        e_outQueue.FreeTensor(e_outLocal);
        m_s_outQueue.FreeTensor(m_s_outLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> e_outQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> m_s_outQueue;
    GlobalTensor<T> input;
    GlobalTensor<uint8_t> e_output, m_s_output;
    uint32_t blockElements;
    uint32_t tileNum;
    uint32_t tileLength;
};

__global__ __aicore__ void generate_table(GM_ADDR input, GM_ADDR output, 
                                                     GM_ADDR totalElements) {
    //
}

__global__ __aicore__ void comp(GM_ADDR e_input, GM_ADDR table, GM_ADDR max_bits_length, GM_ADDR compressed, GM_ADDR totalElements) {
    // 获取总元素数
    uint32_t total = *(reinterpret_cast<const uint32_t*>(totalElements));
    uint32_t blockId = GetBlockIdx();
    uint32_t blockNum = GetBlockNum();

    uint32_t perBlock = (total + blockNum - 1) / blockNum;
    uint32_t start = blockId * perBlock;
    if(start >= total) return;
    uint32_t end = min(start + perBlock, total);
    uint32_t blockElements = end - start;

    GlobalTensor<uint8_t> einputGm(e_input + start * sizeof(uint8_t));
    GlobalTensor<uint8_t> tableinputGm(table);
    GlobalTensor<uint8_t> eOutputGm(e_compressed + start * sizeof(uint8_t));
    GlobalTensor<uint8_t> max_bits_lengthOutputGm();

    CompressKernel<uint8_t> op;
    op.Init(einputGm, tableinputGm, max_bits_length, eoutputGm, blockElements);
    op.Process();
}

__global__ __aicore__ void coalesce(GM_ADDR input, GM_ADDR e_output, 
                                                 GM_ADDR m_s_output, GM_ADDR totalElements) {
    //
}

extern "C" void compress(GM_ADDR BLOCK_NUM, nullptr, stream, srcDevice, inputByteSize, compressedDevice, totalCompressedSize) {
    extractbits<<< >>>(GM_ADDR input, GM_ADDR e_output, GM_ADDR m_s_output, GM_ADDR totalElements);
    generate_table<<< >>>();
    comp<<< >>>();
    coalesce<<< >>>();
}

// 注册算子
__attribute__((visibility("default"))) 
void RegisterExtractBitsKernel() {
    KernelRegistrar<extractbits>()
        .Input(GM_TYPE_UINT16)
        .Output(GM_TYPE_UINT8, "e_output")
        .Output(GM_TYPE_UINT8, "m_s_output")
        .Attr("totalElements", REQUIRED_ATTR);
}


