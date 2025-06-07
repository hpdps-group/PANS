#ifndef HEADER_ACLRTLAUNCH_COALESCE_H
#define HEADER_ACLRTLAUNCH_COALESCE_H
#include "acl/acl_base.h"

#ifndef ACLRT_LAUNCH_KERNEL
#define ACLRT_LAUNCH_KERNEL(kernel_func) aclrtlaunch_##kernel_func
#endif

extern "C" uint32_t aclrtlaunch_coalesce(uint32_t blockDim, aclrtStream stream, uint32_t dataBlockNum, void* finalCompressedExp, void* compressedSize, void* compressedSizePrefix, uint32_t totalUncompressedBytes);
#endif
