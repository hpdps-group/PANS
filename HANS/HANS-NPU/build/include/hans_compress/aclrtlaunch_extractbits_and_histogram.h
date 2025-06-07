#ifndef HEADER_ACLRTLAUNCH_EXTRACTBITS_AND_HISTOGRAM_H
#define HEADER_ACLRTLAUNCH_EXTRACTBITS_AND_HISTOGRAM_H
#include "acl/acl_base.h"

#ifndef ACLRT_LAUNCH_KERNEL
#define ACLRT_LAUNCH_KERNEL(kernel_func) aclrtlaunch_##kernel_func
#endif

extern "C" uint32_t aclrtlaunch_extractbits_and_histogram(uint32_t blockDim, aclrtStream stream, uint32_t datablockNum, void* in, void* tempBuffer, void* final, void* histogramDevice, uint32_t totalUncompressedSize);
#endif
