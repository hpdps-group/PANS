#ifndef HEADER_ACLRTLAUNCH_COMP_H
#define HEADER_ACLRTLAUNCH_COMP_H
#include "acl/acl_base.h"

#ifndef ACLRT_LAUNCH_KERNEL
#define ACLRT_LAUNCH_KERNEL(kernel_func) aclrtlaunch_##kernel_func
#endif

extern "C" uint32_t aclrtlaunch_comp(uint32_t blockDim, aclrtStream stream, uint32_t datablockNum, void* tempBuffer, void* final, void* histogramDevice, void* compressedSize, uint32_t totalUncompressedBytes);
#endif
