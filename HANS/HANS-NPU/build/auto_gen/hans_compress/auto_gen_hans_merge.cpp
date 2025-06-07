#ifndef __HANS_MERGE__KERNEL_FUN_H__
#define __HANS_MERGE__KERNEL_FUN_H__

#undef __global__
#define __global__ inline
#define calcprefix calcprefix_origin
#define coalesce coalesce_origin
#include "/root/yjw/HANS/HANS-NPU/hans_merge.cpp"

#undef calcprefix
#undef coalesce
#undef __global__
#if ASCENDC_CPU_DEBUG
#define __global__
#else
#define __global__ __attribute__((cce_kernel))
#endif

#ifndef ONE_CORE_DUMP_SIZE
#define ONE_CORE_DUMP_SIZE 1048576 * 1
#endif

extern "C" __global__ [aicore] void auto_gen_calcprefix_kernel(
uint32_t datablockNum, __attribute__((cce_global)) uint8_t* tilePrefix, __attribute__((cce_global)) uint8_t* compressedSize, __attribute__((cce_global)) uint8_t* compressedSizePrefix, GM_ADDR overflow_status) {
#if defined(HAVE_WORKSPACE)
    GM_ADDR workspace_param;
    GM_ADDR workspace_usr;
#if defined(HAVE_TILING)
    workspace_param = compressedSize;
#else
    workspace_param = compressedSizePrefix;
#endif
    AscendC::SetSysWorkspaceForce(workspace_param);
    workspace_usr = AscendC::GetUserWorkspace(workspace_param);
#if defined(HAVE_TILING)
    compressedSize = workspace_usr;
#else
    compressedSizePrefix = workspace_usr;
#endif
#endif
    calcprefix_origin(datablockNum, tilePrefix, compressedSize, compressedSizePrefix);
#if defined(ASCENDC_DUMP) && defined(ASCENDC_DEBUG)
    AscendC::WriteBackOverflow(overflow_status);
#endif
}

extern "C" __global__ [aicore] void auto_gen_coalesce_kernel(
uint32_t dataBlockNum, __attribute__((cce_global)) uint8_t* finalCompressedExp, __attribute__((cce_global)) uint8_t* compressedSize, __attribute__((cce_global)) uint8_t* compressedSizePrefix, uint32_t totalUncompressedBytes, GM_ADDR overflow_status) {
#if defined(HAVE_WORKSPACE)
    GM_ADDR workspace_param;
    GM_ADDR workspace_usr;
#if defined(HAVE_TILING)
    workspace_param = compressedSizePrefix;
#else
    workspace_param = totalUncompressedBytes;
#endif
    AscendC::SetSysWorkspaceForce(workspace_param);
    workspace_usr = AscendC::GetUserWorkspace(workspace_param);
#if defined(HAVE_TILING)
    compressedSizePrefix = workspace_usr;
#else
    totalUncompressedBytes = workspace_usr;
#endif
#endif
    coalesce_origin(dataBlockNum, finalCompressedExp, compressedSize, compressedSizePrefix, totalUncompressedBytes);
#if defined(ASCENDC_DUMP) && defined(ASCENDC_DEBUG)
    AscendC::WriteBackOverflow(overflow_status);
#endif
}

#endif
#include "inner_interface/inner_kernel_operator_intf.h"
