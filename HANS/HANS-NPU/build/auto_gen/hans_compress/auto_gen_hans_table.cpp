#ifndef __HANS_TABLE__KERNEL_FUN_H__
#define __HANS_TABLE__KERNEL_FUN_H__

#undef __global__
#define __global__ inline
#define MergeHistogram MergeHistogram_origin
#define extractbits_and_histogram extractbits_and_histogram_origin
#include "/root/yjw/HANS/HANS-NPU/hans_table.cpp"

#undef MergeHistogram
#undef extractbits_and_histogram
#undef __global__
#if ASCENDC_CPU_DEBUG
#define __global__
#else
#define __global__ __attribute__((cce_kernel))
#endif

#ifndef ONE_CORE_DUMP_SIZE
#define ONE_CORE_DUMP_SIZE 1024 * 1
#endif

extern "C" __global__ [aicore] void auto_gen_MergeHistogram_kernel(
#if defined ASCENDC_DUMP || defined ASCENDC_TIME_STAMP_ON
GM_ADDR dumpAddr,
#endif
__attribute__((cce_global)) uint8_t* hist_in, __attribute__((cce_global)) uint8_t* table, GM_ADDR overflow_status) {
#if defined ASCENDC_DUMP || defined ASCENDC_TIME_STAMP_ON
    AscendC::InitDump(false, dumpAddr, ONE_CORE_DUMP_SIZE);
#ifdef ASCENDC_TIME_STAMP_ON
    AscendC::PrintTimeStamp(static_cast<uint32_t>(AscendC::TimeStampId::TIME_STAMP_WRAP_INIT_DUMP));
#endif
#endif

#if defined(HAVE_WORKSPACE)
    GM_ADDR workspace_param;
    GM_ADDR workspace_usr;
#if defined(HAVE_TILING)
    workspace_param = hist_in;
#else
    workspace_param = table;
#endif
    AscendC::SetSysWorkspaceForce(workspace_param);
    workspace_usr = AscendC::GetUserWorkspace(workspace_param);
#if defined(HAVE_TILING)
    hist_in = workspace_usr;
#else
    table = workspace_usr;
#endif
#endif
    MergeHistogram_origin(hist_in, table);
#if defined(ASCENDC_DUMP) && defined(ASCENDC_DEBUG)
    AscendC::WriteBackOverflow(overflow_status);
#endif
}

extern "C" __global__ [aicore] void auto_gen_extractbits_and_histogram_kernel(
#if defined ASCENDC_DUMP || defined ASCENDC_TIME_STAMP_ON
GM_ADDR dumpAddr,
#endif
uint32_t datablockNum, __attribute__((cce_global)) uint8_t* in, __attribute__((cce_global)) uint8_t* tempBuffer, __attribute__((cce_global)) uint8_t* final, __attribute__((cce_global)) uint8_t* histogramDevice, uint32_t totalUncompressedSize, GM_ADDR overflow_status) {
#if defined ASCENDC_DUMP || defined ASCENDC_TIME_STAMP_ON
    AscendC::InitDump(false, dumpAddr, ONE_CORE_DUMP_SIZE);
#ifdef ASCENDC_TIME_STAMP_ON
    AscendC::PrintTimeStamp(static_cast<uint32_t>(AscendC::TimeStampId::TIME_STAMP_WRAP_INIT_DUMP));
#endif
#endif

#if defined(HAVE_WORKSPACE)
    GM_ADDR workspace_param;
    GM_ADDR workspace_usr;
#if defined(HAVE_TILING)
    workspace_param = histogramDevice;
#else
    workspace_param = totalUncompressedSize;
#endif
    AscendC::SetSysWorkspaceForce(workspace_param);
    workspace_usr = AscendC::GetUserWorkspace(workspace_param);
#if defined(HAVE_TILING)
    histogramDevice = workspace_usr;
#else
    totalUncompressedSize = workspace_usr;
#endif
#endif
    extractbits_and_histogram_origin(datablockNum, in, tempBuffer, final, histogramDevice, totalUncompressedSize);
#if defined(ASCENDC_DUMP) && defined(ASCENDC_DEBUG)
    AscendC::WriteBackOverflow(overflow_status);
#endif
}

#endif
#include "inner_interface/inner_kernel_operator_intf.h"
