#include "kernel_operator.h"
#include "hans_utils.h"

using namespace AscendC;

constexpr uint32_t DATA_BLOCK_BYTE_NUM = 4096;
constexpr int32_t BUFFER_NUM = 2; 
constexpr int32_t BLOCK_NUM = 256;
constexpr uint32_t HISTOGRAM_BINS = 256;
constexpr uint32_t HANDLE_NUM_PER = 32; // 算子每次向量化处理32单位的数据量，直方图计算中每个block生成32个temp_table，一共32 * 4 * 256 = 32KB
constexpr uint32_t TILE_LEN = 32; // 每个Tile处理32个单元(单元指输入数据的类型)
constexpr uint32_t TILE_NUM = 32;
// constexpr uint32_t HISTOGRAM_ADD_NUM = 5;

template<typename T>
class ExtractBits1Kernel {
public:
    __aicore__ inline ExtractBits1Kernel() {} // 切分数据，分离指数位，同时进行histogram统计
    // 输入：uint16_t数组(两两组成一个int32_t)
    // 输出：指数数组（int32_t），尾数+sign数组（uint8_t)，histogram统计数组（int32_t）

    __aicore__ inline void Init(__gm__ uint8_t* in, 
                                __gm__ uint8_t* e_out, 
                                __gm__ uint8_t* m_s_out, 
                                __gm__ uint8_t* hist_out, 
                                uint32_t totalElements) {
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

        // this->blockElements = inblockElements ;
        // / sizeof(T);//假设block处理的数据量为32字节的倍数,inblockElements以字节为单位，blockElements以sizeof(T)字节为单位
        this->tileNum = (blockElements + TILE_LEN - 1) / (TILE_LEN); //每次处理32个int32_t的数据量，计算处理的次数

        // assert(tileNum == 4);
        // 初始化Pipe缓冲区，每个Tile大小TILE_LEN
        pipe.InitBuffer(inQueue, BUFFER_NUM, TILE_LEN * sizeof(T));
        pipe.InitBuffer(e_outQueue0, BUFFER_NUM, TILE_LEN * sizeof(uint32_t));
        pipe.InitBuffer(e_outQueue1, BUFFER_NUM, TILE_LEN * sizeof(uint32_t));
        pipe.InitBuffer(m_s_outQueue, BUFFER_NUM, TILE_LEN / 2 * sizeof(uint32_t));
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
            CopyIn0(offset, len);
            Compute0(len, mask0_tensor, mask1_tensor, mask2_tensor, mask3_tensor, mask4_tensor, all_one, offset_tensor, histogram, tempLocal0, tempLocal1, tempLocal2, tempLocal3, histTensor0, histTensor1);
            CopyOut0(offset, len);
        }
        MergeHistogram(histogram);// 合并32个temp直方图为最终的一个

        uint32_t blockId = GetBlockIdx();
        uint32_t blockNum = GetBlockNum();

        for(uint32_t i = startoffset; i > 0; i << 1){
            uint32_t offset = HISTOGRAM_BINS * i;
            if(blockId < i){
                CopyIn1(offset);
                Compute1();
                CopyOut1(offset);
            }
        }
    }

private:
    __aicore__ inline void CopyIn0(uint32_t offset, uint32_t len) {
        LocalTensor<T> inLocal = inQueue.AllocTensor<T>();
        
        // 拷贝当前Tile数据到Local
        DataCopy(inLocal, input[offset], len);
        inQueue.EnQue(inLocal);
    }

    __aicore__ inline void CopyIn1(uint32_t offset, uint32_t len) {
        LocalTensor<T> inLocal = inQueue.AllocTensor<T>();
        
        // 拷贝当前Tile数据到Local
        DataCopy(inLocal, input[offset], len);
        inQueue.EnQue(inLocal);
    }

    __aicore__ inline void Compute0(uint32_t len,
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
        LocalTensor<uint32_t> e_outLocal = e_outQueue0.AllocTensor<uint32_t>();
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
        e_outQueue0.EnQue(e_outLocal0);
        m_s_outQueue.EnQue(m_s_outLocal);
    }

    __aicore__ inline void Compute0(){

    }

    __aicore__ inline void CopyOut0(uint32_t offset, uint32_t len) {
        LocalTensor<uint32_t> e_outLocal = e_outQueue.DeQue<uint32_t>();
        LocalTensor<uint32_t> m_s_outLocal = m_s_outQueue.DeQue<uint32_t>();

        // 将结果拷贝回Global Memory
        DataCopy(e_output0[offset], e_outLocal, len);
        DataCopy(m_s_output[offset], m_s_outLocal, len / 2);

        e_outQueue0.FreeTensor(e_outLocal);
        m_s_outQueue.FreeTensor(m_s_outLocal);
    }

    __aicore__ inline void CopyOut0(){

    }

    __aicore__ inline void MergeHistogram(LocalTensor<int32_t>& histogram) {
        
        for (uint32_t i = 1; i < TILE_LEN; ++i) {
            for (uint32_t bin = 0; bin < HISTOGRAM_BINS; ++bin) {
                histogram(bin) += histogram(i * HISTOGRAM_BINS + bin);
            }
        }
        DataCopy(hist_output, histogram, HISTOGRAM_BINS);
    }

private:
    TPipe pipe;
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
    TBuf<TPosition::VECCALC> offset;
    TBuf<TPosition::VECCALC> one;

    GlobalTensor<T> input;
    GlobalTensor<uint32_t> e_output;
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

// mask0 : 0b 00000000 00000000 00000000 00000000
// mask1 : 0b 00000000 00000000 00000000 00000001 
// mask2 : 0b 00000000 00000000 00000000 11111111
// mask3 : 0b 00000000 00000000 11111111 11111111

// write_mask_8_0 
// 00000000 00000000 00000000 00000000 00000000 00000000 00000000 11111111
// 00000000 00000000 00000000 00000000 00000000 00000000 00000000 11111111
// 00000000 00000000 00000000 00000000 00000000 00000000 00000000 11111111
// 00000000 00000000 00000000 00000000 00000000 00000000 00000000 11111111

// write_mask_8_1 
// 00000000 00000000 00000000 00000000 00000000 00000000 11111111 00000000
// 00000000 00000000 00000000 00000000 00000000 00000000 11111111 00000000
// 00000000 00000000 00000000 00000000 00000000 00000000 11111111 00000000
// 00000000 00000000 00000000 00000000 00000000 00000000 11111111 00000000

// write_mask_8_2
// 00000000 00000000 00000000 00000000 00000000 11111111 00000000 00000000
// 00000000 00000000 00000000 00000000 00000000 11111111 00000000 00000000
// 00000000 00000000 00000000 00000000 00000000 11111111 00000000 00000000
// 00000000 00000000 00000000 00000000 00000000 11111111 00000000 00000000

// write_mask_8_3
// 00000000 00000000 00000000 00000000 11111111 00000000 00000000 00000000
// 00000000 00000000 00000000 00000000 11111111 00000000 00000000 00000000
// 00000000 00000000 00000000 00000000 11111111 00000000 00000000 00000000
// 00000000 00000000 00000000 00000000 11111111 00000000 00000000 00000000

// write_mask_8_4
// 00000000 00000000 00000000 11111111 00000000 00000000 00000000 00000000
// 00000000 00000000 00000000 11111111 00000000 00000000 00000000 00000000
// 00000000 00000000 00000000 11111111 00000000 00000000 00000000 00000000
// 00000000 00000000 00000000 11111111 00000000 00000000 00000000 00000000

// write_mask_8_5
// 00000000 00000000 11111111 00000000 00000000 00000000 00000000 00000000
// 00000000 00000000 11111111 00000000 00000000 00000000 00000000 00000000
// 00000000 00000000 11111111 00000000 00000000 00000000 00000000 00000000
// 00000000 00000000 11111111 00000000 00000000 00000000 00000000 00000000

// write_mask_8_6
// 00000000 11111111 00000000 00000000 00000000 00000000 00000000 00000000
// 00000000 11111111 00000000 00000000 00000000 00000000 00000000 00000000
// 00000000 11111111 00000000 00000000 00000000 00000000 00000000 00000000
// 00000000 11111111 00000000 00000000 00000000 00000000 00000000 00000000

// write_mask_8_7
// 11111111 00000000 00000000 00000000 00000000 00000000 00000000 00000000
// 11111111 00000000 00000000 00000000 00000000 00000000 00000000 00000000
// 11111111 00000000 00000000 00000000 00000000 00000000 00000000 00000000
// 11111111 00000000 00000000 00000000 00000000 00000000 00000000 00000000

// write_mask_7_0 
// 00000000 00000000 00000000 00000000 00000000 00000000 01111111
// 00000000 00000000 00000000 00000000 00000000 00000000 01111111
// 00000000 00000000 00000000 00000000 00000000 00000000 01111111
// 00000000 00000000 00000000 00000000 00000000 00000000 01111111

// write_mask_7_1 
// 00000000 00000000 00000000 00000000 00000000 00111111 10000000
// 00000000 00000000 00000000 00000000 00000000 00111111 10000000
// 00000000 00000000 00000000 00000000 00000000 00111111 10000000
// 00000000 00000000 00000000 00000000 00000000 00111111 10000000

// write_mask_7_2
// 00000000 00000000 00000000 00000000 00011111 11000000 00000000
// 00000000 00000000 00000000 00000000 00011111 11000000 00000000
// 00000000 00000000 00000000 00000000 00011111 11000000 00000000
// 00000000 00000000 00000000 00000000 00011111 11000000 00000000

// write_mask_7_3
// 00000000 00000000 00000000 00001111 11100000 00000000 00000000
// 00000000 00000000 00000000 00001111 11100000 00000000 00000000
// 00000000 00000000 00000000 00001111 11100000 00000000 00000000
// 00000000 00000000 00000000 00001111 11100000 00000000 00000000

// write_mask_7_4
// 00000000 00000000 00000111 11110000 00000000 00000000 00000000
// 00000000 00000000 00000111 11110000 00000000 00000000 00000000
// 00000000 00000000 00000111 11110000 00000000 00000000 00000000
// 00000000 00000000 00000111 11110000 00000000 00000000 00000000

// write_mask_7_5
// 00000000 00000011 11111000 00000000 00000000 00000000 00000000
// 00000000 00000011 11111000 00000000 00000000 00000000 00000000
// 00000000 00000011 11111000 00000000 00000000 00000000 00000000
// 00000000 00000011 11111000 00000000 00000000 00000000 00000000

// write_mask_7_6
// 00000001 11111100 00000000 00000000 00000000 00000000 00000000
// 00000001 11111100 00000000 00000000 00000000 00000000 00000000
// 00000001 11111100 00000000 00000000 00000000 00000000 00000000
// 00000001 11111100 00000000 00000000 00000000 00000000 00000000

// write_mask_7_7
// 11111110 00000000 00000000 00000000 00000000 00000000 00000000 
// 11111110 00000000 00000000 00000000 00000000 00000000 00000000 
// 11111110 00000000 00000000 00000000 00000000 00000000 00000000 
// 11111110 00000000 00000000 00000000 00000000 00000000 00000000 
template<typename T>
class CompressKernel {
public:
    __aicore__ inline CompressKernel() {}
    // 输入：指数数组（uint8_t），table(uint8_t)，
    // 输出：max_bit_length数组（3bits * block_num），码字（max_bit_length * blockSize）

    __aicore__ inline void Init(__gm__ uint8_t* tempBuffer, //e_input
                                __gm__ uint8_t* final, //output
                                __gm__ uint8_t* histogramDevice, //table_input
                                __gm__ uint8_t* compressedSize,
                                uint32_t totalUncompressedBytes) {
        this->blockId = GetBlockIdx();
        this->blockNum = GetBlockNum();

        e_input.SetGlobalBuffer((__gm__ uint32_t*)(tempBuffer));
        table_input.SetGlobalBuffer((__gm__ uint32_t*)(histogramDevice));
        mbl_output.SetGlobalBuffer((__gm__ uint8_t*)(final + 16 + HISTOGRAM_BINS));
        e_output.SetGlobalBuffer((__gm__ uint8_t*)(final + 16 + HISTOGRAM_BINS + 32 * blockNum + 2048 * blockNum));

        // 初始化Pipe缓冲区，每个Tile大小TILE_LEN
        pipe.InitBuffer(inQueue, BUFFER_NUM, TILE_LEN * sizeof(T));
        // pipe.InitBuffer(mbl_outQueue, BUFFER_NUM, TILE_LEN * sizeof(uint8_t));
        pipe.InitBuffer(e_outQueue, BUFFER_NUM, TILE_LEN * sizeof(uint8_t));
    }

    __aicore__ inline void Process() {
        pipe.InitBuffer(table, HISTOGRAM_BINS * sizeof(uint32_t));
        LocalTensor<uint32_t> tableLocal = table.Get<uint32_t>();
        DataCopy(tableLocal, table_input, HISTOGRAM_BINS);

        pipe.InitBuffer(bits_length, HISTOGRAM_BINS * sizeof(uint32_t));
        LocalTensor<uint32_t> blLocal = bits_length.Get<uint32_t>();
        DataCopy();

        pipe.InitBuffer(write_byte_offset, TILE_LEN * sizeof(uint32_t));
        LocalTensor<uint32_t> byteoffsetLocal = write_byte_offset.Get<uint32_t>();
        DataCopy();

        pipe.InitBuffer(write_bits_offset, TILE_LEN * sizeof(uint32_t));
        LocalTensor<uint32_t> bitsoffsetLocal = write_bits_offset.Get<uint32_t>();
        DataCopy(); 

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

        for(uint32_t i = blockId; i < blockNum; i += blockNum){
            uint32_t offset0 = i * DATA_BLOCK_BYTE_NUM;
            end = min((int)offset0 + DATA_BLOCK_BYTE_NUM, (int)totalUncompressedBytes);
            this->blockDataBytesSize = end - offset0;
            uint32_t compressedSize = 0;
            for(uint32_t tileIdx = 0; tileIdx < TILE_NUM; ++tileIdx){
                uint32_t offset1 = tileIdx * TILE_LEN;
                uint32_t len = min(TILE_LEN, blockDataBytesSize - offset1);
                CopyIn(offset0 + offset1, len);
                Compute(tileIdx, compressedSize, tableLocal, blLocal, byteoffsetLocal, bitsoffsetLocal, tempLocal0, tempLocal1, tempLocal2, tempLocal3, mblLocal);
                //当输出块到了32字节就copy到GM
                uint32_t offset2 = ;
                CopyOut(offset0 + offset1, len);
            }
            DataCopy(mbl_output, mblLocal, TILE_NUM);// 每次DataCopy的数据是32字节的倍数
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t len) {
        LocalTensor<T> inLocal = inQueue.AllocTensor<T>();
        // 拷贝当前Tile数据到Local
        DataCopy(e_inLocal, e_input[offset], len);
        inQueue.EnQue(e_inLocal);
    }

    __aicore__ inline void Merge(uint32_t max_bits_length, LocalTensor<uint32_t>& encodedData, LocalTensor<uint16_t>& e_outLocal, LocalTensor<uint32_t>& mergeLocal0, LocalTensor<uint32_t>& mergeLocal1, LocalTensor<uint32_t>& mergeLocal2){
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
            e_outLocal(9) = (uint16_t)mergeLocal(16);
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

            e_outLocal(8) = (uint16_t)mergeLocal(0);
            e_outLocal(9) = (uint16_t)mergeLocal(8);
            e_outLocal(10) = (uint16_t)mergeLocal(16);
            e_outLocal(11) = (uint16_t)mergeLocal(24);
        }
        else if(){
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

            e_outLocal(8) = (uint16_t)mergeLocal(0);
            e_outLocal(9) = (uint16_t)mergeLocal(8);
            e_outLocal(10) = (uint16_t)mergeLocal(16);
            e_outLocal(11) = (uint16_t)mergeLocal(24);

            ShiftRight(mergeLocal0, mergeLocal2, (uint32_t)16, TILE_LEN);
            ShiftLeft(mergeLocal1, mergeLocal0, (uint16_t)4, TILE_LEN);
            Or(mergeLocal2, mergeLocal0, mergeLocal1[8], TILE_LEN * 2);//长度为16

            e_outLocal(12) = (uint16_t)mergeLocal(0);
            e_outLocal(13) = (uint16_t)mergeLocal(16);  
        }
        else if(){
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

    __aicore__ inline void Compute(uint32_t tileIdx,
                                   uint32_t& compressedSize,
                                   LocalTensor<uint32_t>& tableLocal,
                                   LocalTensor<uint32_t>& blLocal,
                                   LocalTensor<uint32_t>& byteoffsetLocal,
                                   LocalTensor<uint32_t>& bitsoffsetLocal,
                                   LocalTensor<uint32_t>& tempLocal0,
                                   LocalTensor<uint32_t>& tempLocal1,
                                   LocalTensor<uint32_t>& tempLocal2,
                                   LocalTensor<uint32_t>& tempLocal3
    ) {

        LocalTensor<T> e_inLocal = e_inQueue.DeQue<T>();
        LocalTensor<uint8_t> mbl_outLocal = mbl_outQueue.AllocTensor<uint8_t>();
        LocalTensor<uint8_t> e_outLocal = e_outQueue.AllocTensor<uint8_t>();

        uint32_t mblSum = 0;

        And(tempLocal0, e_inLocal, mask2_tensor, len * 2);
        Gather(tempLocal1, table, tempLocal0, (uint32_t)0, len);//gather编码表
        Gather(tempLocal2, blLocal, tempLocal1, (uint32_t)0, len);//gather比特长度
        //求出最大截断bits长度，归约操作
        // for(int i = TILE_LEN / 2; i > 0; i >> 1){
        Max(tempLocal3, tempLocal2, tempLocal2[16], 16);
        Max(tempLocal4, tempLocal3, tempLocal3[8], 8);
        Max(tempLocal2, tempLocal4, tempLocal4[4], 4);
        Max(tempLocal3, tempLocal2, tempLocal2[2], 2);
        Max(tempLocal4, tempLocal3, tempLocal3[1], 1);
        uint32_t max_bits_length0 = tempLocal4(0);
        mblSum += max_bits_length0;
        compressedSize = mblSum << 4;//字节为单位
        Merge();
        // }

        ShiftRight(tempLocal0, e_inLocal, (uint32_t)16, len);
        Gather(tempLocal1, table, tempLocal0, (uint32_t)0, len);//gather编码表
        Gather(tempLocal2, blLocal, tempLocal1, (uint32_t)0, len);//gather比特长度
        Max(tempLocal3, tempLocal2, tempLocal2[16], 16);
        Max(tempLocal4, tempLocal3, tempLocal3[8], 8);
        Max(tempLocal2, tempLocal4, tempLocal4[4], 4);
        Max(tempLocal3, tempLocal2, tempLocal2[2], 2);
        Max(tempLocal4, tempLocal3, tempLocal3[1], 1);
        uint32_t max_bits_length1 = tempLocal4(0);
        mblSum += max_bits_length1;
        compressedSize = mblSum << 4;
        Merge();

        fo

        mbl_outLocal(tile_idx) = (max_bits_length1 << 4) | max_bits_length0;

        inQueue.FreeTensor(e_inLocal);
        e_outQueue.EnQue(e_outLocal);
        // mbl_outQueue.EnQue(mbl_outLocal);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t len) {
        LocalTensor<uint8_t> e_outLocal = ecd_outQueue.DeQue<uint8_t>();
        // 将结果拷贝回Global Memory
        DataCopy(e_output[offset], e_outLocal, len);
        e_outQueue.FreeTensor(e_outLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> e_inQueue;
    TQue<QuePosition::VECOUT, 1> e_outQueue;

    TBuf<AscendC::TPosition::VECCALC> table;
    TBuf<AscendC::TPosition::VECCALC> bits_length;
    TBuf<AscendC::TPosition::VECCALC> write_byte_offset;
    TBuf<AscendC::TPosition::VECCALC> write_bits_offset;
    TBuf<AscendC::TPosition::VECCALC> calcBuf0;
    TBuf<AscendC::TPosition::VECCALC> calcBuf1;
    TBuf<AscendC::TPosition::VECCALC> calcBuf2;
    TBuf<AscendC::TPosition::VECCALC> calcBuf3;
    TBuf<AscendC::TPosition::VECCALC> max_bits_length;
    TBuf<AscendC::TPosition::VECCALC> writeBuf0;
    TBuf<AscendC::TPosition::VECCA:C> writeBuf1;

    GlobalTensor<T> e_input;
    GlobalTensor<T> table_input;
    GlobalTensor<uint8_t> output;

    uint32_t blockDataBytesSize;
    uint32_t tileNum;
    uint32_t tileLength;
    uint32_t blockId;
    uint32_t blockNum;
};

extern "C" __global__ __aicore__ void kernel_compress(GM_ADDR inGm, GM_ADDR eGm0, GM_ADDR eGm1, GM_ADDR msGm, GM_ADDR hist, uint32_t length)
{
    CompressKernel<uint32_t> op; 
    op.Init(inGm, eGm0, eGm1, msGm, hist, length);
    op.Process();
}

void compress_do(uint32_t blockDim, void *stream, uint8_t *inGm, uint8_t *eGm0, uint8_t *eGm1, uint8_t *msGm, uint8_t* hist, uint32_t length)
{
    kernel_ExtractBits1<<<blockDim, nullptr, stream>>>(inGm, eGm0, eGm1, msGm, hist, length);
}


template<typename T>
class CoalesceKernel {
public:
    __aicore__ inline PrefixKernel() {} // 生成数据头，紧缩码字
    // 输入：max_bit_length数组（3bits * block_num），码字（max_bit_length * blockSize）
    // 输出：一整块连续的压缩块，压缩块的大小

    __aicore__ inline void Init(__gm__ uint8_t* compressedSize,
                                __gm__ uint8_t* compressedSizePrefix,
                                uint32_t blockNum
                                
    ) {
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

template<typename T>
class CoalesceKernel {
public:
    __aicore__ inline CoalesceKernel() {} // 生成数据头，紧缩码字
    // 输入：max_bit_length数组（3bits * block_num），码字（max_bit_length * blockSize）
    // 输出：一整块连续的压缩块，压缩块的大小

    __aicore__ inline void Init(uint32_t dataBlockNum,
                                __gm__ uint8_t* tempBuffer1, //e_input
                                __gm__ uint8_t* finalCompressedExp, //output
                                __gm__ uint8_t* compressedSize,
                                __gm__ uint8_t* compressedSizePrefix,
                                uint32_t totalUncompressedBytes) {
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
    // TPipe pipe;
    TQueBind<TPosition::VECIN, TPosition::VECOUT, BUFFER_NUM> queBind;

    GlobalTensor<T> input;//输入每个数据块压缩后的GM地址
    GlobalTensor<T> output;//输出每个数据块压缩后的GM地址
    GlobalTensor<T> compressedSize;
    GlobalTensor<T> compressedSizePrefix;

    uint32_t blockId;
    uint32_t blockNum;
    uint32_t dataBlockNum;
};

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

extern "C" void compress(uint32_t blockNum, nullptr, void* stream, uint8_t* srcDevice, uint8_t* tempBuffer, uint8_t* final, int32_t* histogramDevice, uint32_t* compressedSize, uint32_t* compressedSizePrefix, uint32_t totalCompressedSize) {
    TPipe pipe;
    extractbits_and_histogram<<<blockNum, nullptr, stream>>>(srcDevice, tempBuffer, final);//提取字节并计算直方图
    generate_table(histogramDevice, final + 16);//排序后table(uint8_t数组)直接写进final区域，uint32_t的histogram用于压缩
    comp<<<blockNum, nullptr, stream>>>(tempBuffer, final, reinterpret_cast<uint8_t*>(histogramDevice), reinterpret_cast<uint8_t*>(compressedSize));//压缩函数
    calcprefix<<<blockNum, nullptr, stream>>>(reinterpret_cast<uint8_t*>(compressedSize), reinterpret_cast<uint8_t*>(compressedSizePrefix));//计算前缀和，用于后续块合并，字节为单位，
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


