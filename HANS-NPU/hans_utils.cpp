/**
 * @file hello_world.cpp
 *
 * Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include "kernel_operator.h"

class Vec_Add {
public:
    __aicore__ inline Ved_Add() {}

    __aicore__ inline void Init(__gm__ uint8_t *x, __gm__ uint8_t *y, __gm__ uint8_t *offset)
    {
        x_gm.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t *>(x));
        y_gm.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t *>(y));
        offset_gm.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(offset));

        uint32_t len = 128;
        bufferLen = len;
        tpipe.InitBuffer(vecIn, 2, bufferLen * sizeof(uint16_t));
        tpipe.InitBuffer(vecOffset, 2, 8 * sizeof(uint32_t));
        tpipe.InitBuffer(vecOut, 2, bufferLen * sizeof(uint16_t));
    }

    __aicore__ inline void CopyIn(uint32_t index)
    {
        auto x_buf = vecIn.AllocTensor<uint16_t>();
        auto offset_buf = vecOffset.AllocTensor<uint32_t>();
        AscendC::DataCopy(x_buf, x_gm[index * bufferLen], bufferLen);
        AscendC::DataCopy(offset_buf, offset_gm[0], 8);
        vecIn.EnQue(x_buf);
        vecOffset.EnQue(offset_buf);
    }

    __aicore__ inline void CopyOut(uint32_t index)
    {
        auto y_buf = vecOut.DeQue<uint16_t>();
        AscendC::DataCopy(y_gm[index * bufferLen], y_buf, bufferLen);
        vecOut.FreeTensor(y_buf);
    }

    __aicore__ inline void Compute()
    {
        auto x_buf = vecIn.DeQue<uint16_t>();
        auto offset_buf = vecOffset.DeQue<uint32_t>();
        auto y_buf = vecOut.AllocTensor<uint16_t>();
        AscendC::GatherRepeatParams params{1, 8};
        uint8_t repeatTime = bufferLen * sizeof(uint16_t) / 256;
        AscendC::Gatherb<uint16_t>(y_buf, x_buf, offset_buf, repeatTime, params);
        vecIn.FreeTensor(x_buf);
        vecOffset.FreeTensor(offset_buf);
        vecOut.EnQue(y_buf);
    }

    __aicore__ inline void Process()
    {
        for (int i = 0; i < 1; i++) {
            CopyIn(i);
            Compute();
            CopyOut(i);
        }
    }

private:
    AscendC::GlobalTensor<uint16_t> x_gm;
    AscendC::GlobalTensor<uint16_t> y_gm;
    AscendC::GlobalTensor<uint32_t> offset_gm;

    AscendC::TPipe tpipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 2> vecIn;
    AscendC::TQue<AscendC::TPosition::VECIN, 2> vecOffset;
    AscendC::TQue<AscendC::TPosition::VECOUT, 2> vecOut;

    uint32_t bufferLen = 0;
};

extern "C" __global__ __aicore__ void Vec_Add(__gm__ uint8_t *x, __gm__ uint8_t *y, __gm__ uint8_t *z)
{
    VgatherbCase op;
    op.Init(x, y, z); // z = x + y
    op.Process();
}

class VgatherbCase {
public:
    __aicore__ inline VgatherbCase() {}

    __aicore__ inline void Init(__gm__ uint8_t *x, __gm__ uint8_t *y, __gm__ uint8_t *offset)
    {
        x_gm.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t *>(x));
        y_gm.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t *>(y));
        offset_gm.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(offset));

        uint32_t len = 128;
        bufferLen = len;
        tpipe.InitBuffer(vecIn, 2, bufferLen * sizeof(uint16_t));
        tpipe.InitBuffer(vecOffset, 2, 8 * sizeof(uint32_t));
        tpipe.InitBuffer(vecOut, 2, bufferLen * sizeof(uint16_t));
    }

    __aicore__ inline void CopyIn(uint32_t index)
    {
        auto x_buf = vecIn.AllocTensor<uint16_t>();
        auto offset_buf = vecOffset.AllocTensor<uint32_t>();
        AscendC::DataCopy(x_buf, x_gm[index * bufferLen], bufferLen);
        AscendC::DataCopy(offset_buf, offset_gm[0], 8);
        vecIn.EnQue(x_buf);
        vecOffset.EnQue(offset_buf);
    }

    __aicore__ inline void CopyOut(uint32_t index)
    {
        auto y_buf = vecOut.DeQue<uint16_t>();
        AscendC::DataCopy(y_gm[index * bufferLen], y_buf, bufferLen);
        vecOut.FreeTensor(y_buf);
    }

    __aicore__ inline void Compute()
    {
        auto x_buf = vecIn.DeQue<uint16_t>();
        auto offset_buf = vecOffset.DeQue<uint32_t>();
        auto y_buf = vecOut.AllocTensor<uint16_t>();
        AscendC::GatherRepeatParams params{1, 8};
        uint8_t repeatTime = bufferLen * sizeof(uint16_t) / 256;
        AscendC::Gatherb<uint16_t>(y_buf, x_buf, offset_buf, repeatTime, params);
        vecIn.FreeTensor(x_buf);
        vecOffset.FreeTensor(offset_buf);
        vecOut.EnQue(y_buf);
    }

    __aicore__ inline void Process()
    {
        for (int i = 0; i < 1; i++) {
            CopyIn(i);
            Compute();
            CopyOut(i);
        }
    }

private:
    AscendC::GlobalTensor<uint16_t> x_gm;
    AscendC::GlobalTensor<uint16_t> y_gm;
    AscendC::GlobalTensor<uint32_t> offset_gm;

    AscendC::TPipe tpipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 2> vecIn;
    AscendC::TQue<AscendC::TPosition::VECIN, 2> vecOffset;
    AscendC::TQue<AscendC::TPosition::VECOUT, 2> vecOut;

    uint32_t bufferLen = 0;
};

extern "C" __global__ __aicore__ void vgatherb_core(__gm__ uint8_t *x, __gm__ uint8_t *y, __gm__ uint8_t *offset)
{
    VgatherbCase op;
    op.Init(x, y, offset);
    op.Process();
}

template <typename T>
class ScatterTest {
public:
    __aicore__ inline ScatterTest() {}
    __aicore__ inline void Init(__gm__ uint8_t* dstGm, __gm__ uint8_t* srcGm,
        __gm__ uint8_t* dstOffsetGm, const uint32_t count)
    {
        m_elementCount = count;
        m_dstGlobal.SetGlobalBuffer((__gm__ T*)dstGm);
        m_srcGlobal.SetGlobalBuffer((__gm__ T*)srcGm);
        m_dstOffsetGlobal.SetGlobalBuffer((__gm__ uint32_t*)dstOffsetGm);
        m_pipe.InitBuffer(m_queIn, 2, m_elementCount * sizeof(uint32_t));
        m_pipe.InitBuffer(m_queOut, 1, m_elementCount * sizeof(uint32_t));
    }
    __aicore__ inline void Process()
    {
        CopyIn();
        Compute();
        CopyOut();
    }
private:
    __aicore__ inline void CopyIn()
    {
        AscendC::LocalTensor<T> srcLocal = m_queIn.AllocTensor<T>();
        AscendC::DataCopy(srcLocal, m_srcGlobal, m_elementCount);
        m_queIn.EnQue(srcLocal);
        AscendC::LocalTensor<uint32_t> dstOffsetLocal = m_queIn.AllocTensor<uint32_t>();
        AscendC::DataCopy(dstOffsetLocal, m_dstOffsetGlobal, m_elementCount);
        m_queIn.EnQue(dstOffsetLocal);
    }
    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<T> srcLocal = m_queIn.DeQue<T>();
        AscendC::LocalTensor<uint32_t> dstOffsetLocal = m_queIn.DeQue<uint32_t>();
        AscendC::LocalTensor<T> dstLocal = m_queOut.AllocTensor<T>();
        dstLocal.SetSize(m_elementCount);
        AscendC::Scatter(dstLocal, srcLocal, dstOffsetLocal, (uint32_t)0, m_elementCount);
        m_queIn.FreeTensor(srcLocal);
        m_queIn.FreeTensor(dstOffsetLocal);
        m_queOut.EnQue(dstLocal);
    }
    __aicore__ inline void CopyOut()
    {
        AscendC::LocalTensor<T> dstLocal = m_queOut.DeQue<T>();
        AscendC::DataCopy(m_dstGlobal, dstLocal, m_elementCount);
        m_queOut.FreeTensor(dstLocal);
    }
private:
    AscendC::TPipe m_pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> m_queCalc;
    AscendC::GlobalTensor<T> m_valueGlobal;
    uint32_t m_concatRepeatTimes;
    uint32_t m_sortRepeatTimes;
    uint32_t m_extractRepeatTimes;
    uint32_t m_elementCount;
    AscendC::GlobalTensor<uint32_t> m_dstOffsetGlobal;
    AscendC::GlobalTensor<T> m_srcGlobal;
    AscendC::GlobalTensor<T> m_dstGlobal;
    AscendC::TQue<AscendC::TPosition::VECIN, 2> m_queIn;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> m_queOut;
}; // class ScatterTest

#define KERNEL_SCATTER(T, count)                                                                    \
    extern "C" __global__ __aicore__ void kernel_scatter_##T##_##count(GM_ADDR dstGm, GM_ADDR srcGm,\
        GM_ADDR dstOffsetGm)                                                                        \
    {                                                                                               \
        ScatterTest<T> op;                                                                          \
        op.Init(dstGm, srcGm, dstOffsetGm, count);                                                  \
        op.Process();                                                                               \
    }