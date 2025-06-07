#ifndef __HANS_COMPRESS__KERNEL_FUN_H__
#define __HANS_COMPRESS__KERNEL_FUN_H__

#undef __global__
#define __global__ inline
#define comp comp_origin
#include "/root/yjw/HANS/HANS-NPU/hans_compress.cpp"

#undef comp
#undef __global__
#if ASCENDC_CPU_DEBUG
#define __global__
#else
#define __global__ __attribute__((cce_kernel))
#endif

#ifndef ONE_CORE_DUMP_SIZE
#define ONE_CORE_DUMP_SIZE 1048576 * 1
#endif

extern "C" __global__ [aicore] void auto_gen_comp_kernel(
uint32_t datablockNum, __attribute__((cce_global)) uint8_t* tempBuffer, __attribute__((cce_global)) uint8_t* final, __attribute__((cce_global)) uint8_t* histogramDevice, __attribute__((cce_global)) uint8_t* compressedSize, uint32_t totalUncompressedBytes, GM_ADDR overflow_status) {
#if defined(HAVE_WORKSPACE)
    GM_ADDR workspace_param;
    GM_ADDR workspace_usr;
#if defined(HAVE_TILING)
    workspace_param = compressedSize;
#else
    workspace_param = totalUncompressedBytes;
#endif
    AscendC::SetSysWorkspaceForce(workspace_param);
    workspace_usr = AscendC::GetUserWorkspace(workspace_param);
#if defined(HAVE_TILING)
    compressedSize = workspace_usr;
#else
    totalUncompressedBytes = workspace_usr;
#endif
#endif
    comp_origin(datablockNum, tempBuffer, final, histogramDevice, compressedSize, totalUncompressedBytes);
#if defined(ASCENDC_DUMP) && defined(ASCENDC_DEBUG)
    AscendC::WriteBackOverflow(overflow_status);
#endif
}

#endif
#include "inner_interface/inner_kernel_operator_intf.h"
