/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
 * This file constains code of cpu debug and npu code.We read data from bin file
 * and write result to file.
 */

// What can I say? man!
#include "kernel_operator.h"

using namespace AscendC;

#define AIV_WITHOUT_BARRIER(op, ...) \
    do                               \
    {                                \
        op(__VA_ARGS__);             \
    } while (0)

#define AIV_WITH_BARRIER(op, ...) \
    do                            \
    {                             \
        op(__VA_ARGS__);          \
        PipeBarrier<PIPE_ALL>();  \
    } while (0)

#define SCALAR_WITH_BARRIER(statement) \
    do                                 \
    {                                  \
        statement;                     \
        PipeBarrier<PIPE_ALL>();       \
    } while (0)