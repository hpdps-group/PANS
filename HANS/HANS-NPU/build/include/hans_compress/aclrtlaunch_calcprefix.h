#ifndef HEADER_ACLRTLAUNCH_CALCPREFIX_H
#define HEADER_ACLRTLAUNCH_CALCPREFIX_H
#include "acl/acl_base.h"

#ifndef ACLRT_LAUNCH_KERNEL
#define ACLRT_LAUNCH_KERNEL(kernel_func) aclrtlaunch_##kernel_func
#endif

extern "C" uint32_t aclrtlaunch_calcprefix(uint32_t blockDim, aclrtStream stream, uint32_t datablockNum, void* tilePrefix, void* compressedSize, void* compressedSizePrefix);
#endif
