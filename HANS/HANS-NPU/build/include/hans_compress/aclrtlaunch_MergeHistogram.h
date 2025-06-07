#ifndef HEADER_ACLRTLAUNCH_MERGEHISTOGRAM_H
#define HEADER_ACLRTLAUNCH_MERGEHISTOGRAM_H
#include "acl/acl_base.h"

#ifndef ACLRT_LAUNCH_KERNEL
#define ACLRT_LAUNCH_KERNEL(kernel_func) aclrtlaunch_##kernel_func
#endif

extern "C" uint32_t aclrtlaunch_MergeHistogram(uint32_t blockDim, aclrtStream stream, void* hist_in, void* table);
#endif
