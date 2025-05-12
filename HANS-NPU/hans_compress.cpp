#include "kernel_operator.h"
#include "hans_utils.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2; 
constexpr int32_t BLOCK_NUM = 256;
constexpr uint32_t HISTOGRAM_BINS = 256;
constexpr uint32_t HANDLE_NUM_PER = 32; // 算子每次向量化处理32字节的数据量，直方图计算中每个block生成32个temp_table，一共32 * 4 * 256 = 32KB
constexpr uint32_t TILE_LEN = 32; // 每个Tile处理32字节

template<typename T>
class ExtractBitsKernel {
public:
    __aicore__ inline ExtractBitsKernel() {
        for(int i = 0; i < HISTOGRAM_BINS; i ++){
            histogram_output[i] = 0;
        }
    } // 切分数据，分离指数位，同时进行histogram统计
    // 输入：uint16_t数组
    // 输出：指数数组（uint8_t），尾数+sign数组（uint8_t)，histogram统计数组（uint8_t）

    __aicore__ inline void Init(GlobalTensor<T>& input, GlobalTensor<uint8_t>& e_output, 
                               GlobalTensor<uint8_t>& m_s_output, GlobalTensor<uint8_t>& histogram_output, uint32_t blockElements) {
        this->input = input;
        this->e_output = e_output;
        this->m_s_output = m_s_output;
        this->histogram_output = histogram_output;
        this->blockElements = blockElements;
        this->tileNum = (blockElements + TILE_LEN - 1) / TILE_LEN;

        // 初始化Pipe缓冲区，每个Tile大小TILE_LEN
        pipe.InitBuffer(inQueue, BUFFER_NUM, TILE_LEN * sizeof(T));
        pipe.InitBuffer(e_outQueue, BUFFER_NUM, TILE_LEN * sizeof(uint8_t));
        pipe.InitBuffer(m_s_outQueue, BUFFER_NUM, TILE_LEN * sizeof(uint8_t));
    }

    __aicore__ inline void Process() {
        int32_t loopCount = this->tileNum;
        for (uint32_t tileIdx = 0; tileIdx < loopCount; ++tileIdx) {
            CopyIn(tileIdx);
            Compute(tileIdx);
            CopyOut(tileIdx);
        }
        MergeHistogram();// 合并32个temp直方图为最终的一个
    }

private:
    __aicore__ inline void CopyIn(uint32_t tileIdx) {
        uint32_t offset = tileIdx * TILE_LEN;
        uint32_t copyLen = min(TILE_LEN, blockElements - offset);
        LocalTensor<T> inLocal = inQueue.AllocTensor<T>();
        
        // 拷贝当前Tile数据到Local
        DataCopy(inLocal, input[offset], copyLen);
        inQueue.EnQue(inLocal);
    }

    __aicore__ inline void Compute(uint32_t tileIdx) {
        uint32_t offset = tileIdx * TILE_LEN;
        uint32_t computeLen = min(TILE_LEN, blockElements - offset);
        LocalTensor<T> inLocal = inQueue.DeQue<T>();
        LocalTensor<uint8_t> e_outLocal = e_outQueue.AllocTensor<uint8_t>();
        LocalTensor<uint8_t> m_s_outLocal = m_s_outQueue.AllocTensor<uint8_t>();

        // 处理每个元素，每次提取32个uint16_t
        for (uint32_t i = 0; i < computeLen; ++i) {
            uint16_t val = inLocal.GetValue(i);
            uint32_t extracted_temp = (val << 16) | val; // 两个相同值直接与
            uint8_t extracted_e = (extracted_temp >> 7) & 0xFF;  // 提取高8位
            uint8_t extracted_m_s = (extracted_temp >> 15) & 0xFF;
            histogram_output[extracted_e] ++;
            e_outLocal.SetValue(i, e);
            m_s_outLocal.SetValue(i, m_s);
        }

        inQueue.FreeTensor(inLocal);
        e_outQueue.EnQue(e_outLocal);
        m_s_outQueue.EnQue(m_s_outLocal);
    }

    __aicore__ inline void CopyOut(uint32_t tileIdx) {
        uint32_t offset = tileIdx * TILE_LEN;
        uint32_t copyLen = min(TILE_LEN, blockElements - offset);
        LocalTensor<uint8_t> e_outLocal = e_outQueue.DeQue<uint8_t>();
        LocalTensor<uint8_t> m_s_outLocal = m_s_outQueue.DeQue<uint8_t>();

        // 将结果拷贝回Global Memory
        DataCopy(e_output[offset], e_outLocal, copyLen);
        DataCopy(m_s_output[offset], m_s_outLocal, copyLen);

        e_outQueue.FreeTensor(e_outLocal);
        m_s_outQueue.FreeTensor(m_s_outLocal);
    }

    __aicore__ inline void MergeHistogram() {
        for(int i = 0; i < 5; i ++){
            int offset = 1 << i;
            for(int j = 0; j < TILE_NUM; j += offset){
                Add(histogram_temp[j], histogram_temp[j], histogram_temp[j + offset], HISTOGRAM_BINS);
            }
        }
        DataCopy(histogram_output, histogram_temp[0], HISTOGRAM_BINS);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> e_outQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> m_s_outQueue;
    GlobalTensor<T> input;
    GlobalTensor<uint8_t> e_output;
    GlobalTensor<uint8_t> m_s_output;
    GlobalTensor<uint32_t> histogram_output;
    LocalTensor<uint32_t> histogram_temp[HANDLE_NUM_PER][HISTOGRAM_BINS]; // 这些数据在localMem上
    uint32_t blockElements;
    uint32_t tileNum;
    uint32_t tileLength;
};

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
    GlobalTensor<T> input;
    GlobalTensor<uint8_t> e_output, m_s_output;
    uint32_t blockElements;
    uint32_t tileNum;
    uint32_t tileLength;
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
        ReduceMax(const LocalTensor<T>& dstLocal, const LocalTensor<T>& tempLocal2, const LocalTensor<T>& workLocal, TILE_LEN)


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


