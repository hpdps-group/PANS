#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <dlfcn.h>
#include <securec.h>

#ifndef ASCENDC_DUMP
#define ASCENDC_DUMP 1
#endif

#if defined(ASCENDC_DUMP) && (ASCENDC_DUMP == 0)
    #undef ASCENDC_DUMP
#endif

#ifdef ASCENDC_DUMP
#define ASCENDC_EXCEPTION_DUMP_HEAD 2U

typedef struct rtArgsSizeInfo {
    void *infoAddr;
    uint32_t atomicIndex;
} rtArgsSizeInfo_t;
#endif

static char ascendcErrMsg[1024] = {0};

static void *g_kernel_handle_aiv = nullptr;

struct ascend_kernels {
    uint32_t version;
    uint32_t type_cnt;
    uint32_t aiv_type;
    uint32_t aiv_len;
    uint32_t aiv_file_len;
    uint8_t aiv_buf[426360];
} __ascend_kernel_ascend910b2_hans_compress __attribute__ ((section (".ascend.kernel.ascend910b2.hans_compress"))) = {1,1,1,426360,426360,{0}};

extern "C" {
uint32_t RegisterAscendBinary(const char *fileBuf, size_t fileSize, uint32_t type, void **handle);
uint32_t LaunchAscendKernel(void *handle, const uint64_t key, const uint32_t blockDim, void **args,
                            uint32_t size, const void *stream);
uint32_t GetAscendCoreSyncAddr(void **addr);
int UnregisterAscendBinary(void *hdl);
void StartAscendProf(const char *name, uint64_t *startTime);
void ReportAscendProf(const char *name, uint32_t blockDim, uint32_t taskType, const uint64_t startTime);
bool GetAscendProfStatus();
uint32_t AllocAscendMemDevice(void **devMem, uint64_t size);
uint32_t FreeAscendMemDevice(void *devMem);
bool AscendCheckSoCVersion(const char *socVersion, char* errMsg);
void AscendProfRegister();
uint32_t GetCoreNumForMixVectorCore(uint32_t *aiCoreNum, uint32_t *vectorCoreNum);
uint32_t LaunchAscendKernelForVectorCore(const char *opType, void *handle, const uint64_t key, void **args, uint32_t size,
    const void *stream, bool enbaleProf, uint32_t aicBlockDim, uint32_t aivBlockDim, uint32_t aivBlockDimOffset);
int32_t rtSetExceptionExtInfo(const rtArgsSizeInfo_t * const sizeInfo);

namespace Adx {
    void *AdumpGetSizeInfoAddr(uint32_t space, uint32_t &atomicIndex);
}
}
namespace Adx {

    void AdumpPrintWorkSpace(const void *workSpaceAddr, const size_t dumpWorkSpaceSize,
                            void *stream, const char *opType);
}

    class KernelHandleGradUnregister {
    private:
        KernelHandleGradUnregister() {}

    public:
        KernelHandleGradUnregister(const KernelHandleGradUnregister&) = delete;
        KernelHandleGradUnregister& operator=(const KernelHandleGradUnregister&) = delete;

        static KernelHandleGradUnregister& GetInstance() {
            static KernelHandleGradUnregister instance;
            return instance;
        }
        ~KernelHandleGradUnregister(){
            if (g_kernel_handle_aiv) {
                UnregisterAscendBinary(g_kernel_handle_aiv);
                g_kernel_handle_aiv = nullptr;
            }
        }
    };

static void __register_kernels(void) __attribute__((constructor));
void __register_kernels(void)
{
    const char* compileSocVersion = "ascend910b2";
    uint32_t ret;

    bool checkSocVersion = AscendCheckSoCVersion(compileSocVersion, ascendcErrMsg);
    if (!checkSocVersion) {
        return;
    }
    ret = RegisterAscendBinary(
        (const char *)__ascend_kernel_ascend910b2_hans_compress.aiv_buf,
        __ascend_kernel_ascend910b2_hans_compress.aiv_file_len,
        1,
        &g_kernel_handle_aiv);
    if (ret != 0) {
        printf("RegisterAscendBinary aiv ret %u \n", ret);
    }

    AscendProfRegister();
}

#ifdef ASCENDC_DUMP
static void ascendc_set_exception_dump_info(uint32_t dumpSize)
{
    uint32_t atomicIndex = 0U;
    uint32_t addrNum = 1U;
    void *exceptionDumpAddr = Adx::AdumpGetSizeInfoAddr(addrNum + ASCENDC_EXCEPTION_DUMP_HEAD, atomicIndex);
    if (exceptionDumpAddr == nullptr) {
        printf("Get exceptionDumpAddr is nullptr.\n");
        return;
    }


    uint64_t *sizeInfoAddr = reinterpret_cast<uint64_t *>(exceptionDumpAddr);
    *sizeInfoAddr = static_cast<uint64_t>(atomicIndex);
    sizeInfoAddr++;

    *sizeInfoAddr = static_cast<uint64_t>(1);
    sizeInfoAddr++;

    *sizeInfoAddr = dumpSize * 75;
    constexpr uint64_t workspaceOffset = (4ULL << 56ULL);
    *sizeInfoAddr |= workspaceOffset;

    const rtArgsSizeInfo sizeInfo = {exceptionDumpAddr, atomicIndex};
    int32_t ret = rtSetExceptionExtInfo(&sizeInfo);
    if (ret != 0) {
        printf("rtSetExceptionExtInfo failed, ret = %d.\n", ret);
    }
}
#endif





uint32_t launch_and_profiling_MergeHistogram(uint64_t func_key, uint32_t blockDim, void* stream, void **args, uint32_t size)
{
    uint64_t startTime;
    const char *name = "MergeHistogram";
    bool profStatus = GetAscendProfStatus();
    if (profStatus) {
        StartAscendProf(name, &startTime);
    }
    if (g_kernel_handle_aiv == nullptr) {
        printf("[ERROR] %s\n", ascendcErrMsg);
        return 0;
    }
    uint32_t ret = LaunchAscendKernel(g_kernel_handle_aiv, func_key, blockDim, args, size, stream);
    if (ret != 0) {
        printf("LaunchAscendKernel ret %u\n", ret);
    }
    if (profStatus) {
        ReportAscendProf(name, blockDim, 1, startTime);
    }
    return ret;
}

extern "C" uint32_t aclrtlaunch_MergeHistogram(uint32_t blockDim, void* stream, void* hist_in, void* table)
{
    struct {
    #if defined ASCENDC_DUMP || defined ASCENDC_TIME_STAMP_ON
            void* __ascendc_dump;
    #endif
        alignas(((alignof(void*) + 3) >> 2) << 2) void* hist_in;
        alignas(((alignof(void*) + 3) >> 2) << 2) void* table;
        alignas(((alignof(void*) + 3) >> 2) << 2) void* __ascendc_overflow;
    } __ascendc_args;

    uint32_t __ascendc_ret;
#if defined ASCENDC_DUMP || defined ASCENDC_TIME_STAMP_ON
    constexpr uint32_t __ascendc_one_core_dump_size = 1024;
    AllocAscendMemDevice(&(__ascendc_args.__ascendc_dump), __ascendc_one_core_dump_size * 75);
#endif
    constexpr uint32_t __ascendc_overflow_status_size = 8;
    AllocAscendMemDevice(&(__ascendc_args.__ascendc_overflow), __ascendc_overflow_status_size);
    __ascendc_args.hist_in = hist_in;
    __ascendc_args.table = table;

    ascendc_set_exception_dump_info(__ascendc_one_core_dump_size);
    __ascendc_ret = launch_and_profiling_MergeHistogram(0, blockDim, stream, (void **)&__ascendc_args, sizeof(__ascendc_args));
    KernelHandleGradUnregister::GetInstance();
#if defined ASCENDC_DUMP || defined ASCENDC_TIME_STAMP_ON
    FreeAscendMemDevice(__ascendc_args.__ascendc_dump);
#endif
    FreeAscendMemDevice(__ascendc_args.__ascendc_overflow);
    return __ascendc_ret;
}


uint32_t launch_and_profiling_extractbits_and_histogram(uint64_t func_key, uint32_t blockDim, void* stream, void **args, uint32_t size)
{
    uint64_t startTime;
    const char *name = "extractbits_and_histogram";
    bool profStatus = GetAscendProfStatus();
    if (profStatus) {
        StartAscendProf(name, &startTime);
    }
    if (g_kernel_handle_aiv == nullptr) {
        printf("[ERROR] %s\n", ascendcErrMsg);
        return 0;
    }
    uint32_t ret = LaunchAscendKernel(g_kernel_handle_aiv, func_key, blockDim, args, size, stream);
    if (ret != 0) {
        printf("LaunchAscendKernel ret %u\n", ret);
    }
    if (profStatus) {
        ReportAscendProf(name, blockDim, 1, startTime);
    }
    return ret;
}

extern "C" uint32_t aclrtlaunch_extractbits_and_histogram(uint32_t blockDim, void* stream, uint32_t datablockNum, void* in, void* tempBuffer, void* final, void* histogramDevice, uint32_t totalUncompressedSize)
{
    struct {
    #if defined ASCENDC_DUMP || defined ASCENDC_TIME_STAMP_ON
            void* __ascendc_dump;
    #endif
        alignas(((alignof(uint32_t) + 3) >> 2) << 2) uint32_t datablockNum;
        alignas(((alignof(void*) + 3) >> 2) << 2) void* in;
        alignas(((alignof(void*) + 3) >> 2) << 2) void* tempBuffer;
        alignas(((alignof(void*) + 3) >> 2) << 2) void* final;
        alignas(((alignof(void*) + 3) >> 2) << 2) void* histogramDevice;
        alignas(((alignof(uint32_t) + 3) >> 2) << 2) uint32_t totalUncompressedSize;
        alignas(((alignof(void*) + 3) >> 2) << 2) void* __ascendc_overflow;
    } __ascendc_args;

    uint32_t __ascendc_ret;
#if defined ASCENDC_DUMP || defined ASCENDC_TIME_STAMP_ON
    constexpr uint32_t __ascendc_one_core_dump_size = 1024;
    AllocAscendMemDevice(&(__ascendc_args.__ascendc_dump), __ascendc_one_core_dump_size * 75);
#endif
    constexpr uint32_t __ascendc_overflow_status_size = 8;
    AllocAscendMemDevice(&(__ascendc_args.__ascendc_overflow), __ascendc_overflow_status_size);
    __ascendc_args.datablockNum = datablockNum;
    __ascendc_args.in = in;
    __ascendc_args.tempBuffer = tempBuffer;
    __ascendc_args.final = final;
    __ascendc_args.histogramDevice = histogramDevice;
    __ascendc_args.totalUncompressedSize = totalUncompressedSize;

    ascendc_set_exception_dump_info(__ascendc_one_core_dump_size);
    __ascendc_ret = launch_and_profiling_extractbits_and_histogram(1, blockDim, stream, (void **)&__ascendc_args, sizeof(__ascendc_args));
    KernelHandleGradUnregister::GetInstance();
#if defined ASCENDC_DUMP || defined ASCENDC_TIME_STAMP_ON
    FreeAscendMemDevice(__ascendc_args.__ascendc_dump);
#endif
    FreeAscendMemDevice(__ascendc_args.__ascendc_overflow);
    return __ascendc_ret;
}


uint32_t launch_and_profiling_comp(uint64_t func_key, uint32_t blockDim, void* stream, void **args, uint32_t size)
{
    uint64_t startTime;
    const char *name = "comp";
    bool profStatus = GetAscendProfStatus();
    if (profStatus) {
        StartAscendProf(name, &startTime);
    }
    if (g_kernel_handle_aiv == nullptr) {
        printf("[ERROR] %s\n", ascendcErrMsg);
        return 0;
    }
    uint32_t ret = LaunchAscendKernel(g_kernel_handle_aiv, func_key, blockDim, args, size, stream);
    if (ret != 0) {
        printf("LaunchAscendKernel ret %u\n", ret);
    }
    if (profStatus) {
        ReportAscendProf(name, blockDim, 1, startTime);
    }
    return ret;
}

extern "C" uint32_t aclrtlaunch_comp(uint32_t blockDim, void* stream, uint32_t datablockNum, void* tempBuffer, void* final, void* histogramDevice, void* compressedSize, uint32_t totalUncompressedBytes)
{
    struct {
        alignas(((alignof(uint32_t) + 3) >> 2) << 2) uint32_t datablockNum;
        alignas(((alignof(void*) + 3) >> 2) << 2) void* tempBuffer;
        alignas(((alignof(void*) + 3) >> 2) << 2) void* final;
        alignas(((alignof(void*) + 3) >> 2) << 2) void* histogramDevice;
        alignas(((alignof(void*) + 3) >> 2) << 2) void* compressedSize;
        alignas(((alignof(uint32_t) + 3) >> 2) << 2) uint32_t totalUncompressedBytes;
        alignas(((alignof(void*) + 3) >> 2) << 2) void* __ascendc_overflow;
    } __ascendc_args;

    uint32_t __ascendc_ret;
    constexpr uint32_t __ascendc_overflow_status_size = 8;
    AllocAscendMemDevice(&(__ascendc_args.__ascendc_overflow), __ascendc_overflow_status_size);
    __ascendc_args.datablockNum = datablockNum;
    __ascendc_args.tempBuffer = tempBuffer;
    __ascendc_args.final = final;
    __ascendc_args.histogramDevice = histogramDevice;
    __ascendc_args.compressedSize = compressedSize;
    __ascendc_args.totalUncompressedBytes = totalUncompressedBytes;

    __ascendc_ret = launch_and_profiling_comp(2, blockDim, stream, (void **)&__ascendc_args, sizeof(__ascendc_args));
    KernelHandleGradUnregister::GetInstance();
    FreeAscendMemDevice(__ascendc_args.__ascendc_overflow);
    return __ascendc_ret;
}


uint32_t launch_and_profiling_calcprefix(uint64_t func_key, uint32_t blockDim, void* stream, void **args, uint32_t size)
{
    uint64_t startTime;
    const char *name = "calcprefix";
    bool profStatus = GetAscendProfStatus();
    if (profStatus) {
        StartAscendProf(name, &startTime);
    }
    if (g_kernel_handle_aiv == nullptr) {
        printf("[ERROR] %s\n", ascendcErrMsg);
        return 0;
    }
    uint32_t ret = LaunchAscendKernel(g_kernel_handle_aiv, func_key, blockDim, args, size, stream);
    if (ret != 0) {
        printf("LaunchAscendKernel ret %u\n", ret);
    }
    if (profStatus) {
        ReportAscendProf(name, blockDim, 1, startTime);
    }
    return ret;
}

extern "C" uint32_t aclrtlaunch_calcprefix(uint32_t blockDim, void* stream, uint32_t datablockNum, void* tilePrefix, void* compressedSize, void* compressedSizePrefix)
{
    struct {
        alignas(((alignof(uint32_t) + 3) >> 2) << 2) uint32_t datablockNum;
        alignas(((alignof(void*) + 3) >> 2) << 2) void* tilePrefix;
        alignas(((alignof(void*) + 3) >> 2) << 2) void* compressedSize;
        alignas(((alignof(void*) + 3) >> 2) << 2) void* compressedSizePrefix;
        alignas(((alignof(void*) + 3) >> 2) << 2) void* __ascendc_overflow;
    } __ascendc_args;

    uint32_t __ascendc_ret;
    constexpr uint32_t __ascendc_overflow_status_size = 8;
    AllocAscendMemDevice(&(__ascendc_args.__ascendc_overflow), __ascendc_overflow_status_size);
    __ascendc_args.datablockNum = datablockNum;
    __ascendc_args.tilePrefix = tilePrefix;
    __ascendc_args.compressedSize = compressedSize;
    __ascendc_args.compressedSizePrefix = compressedSizePrefix;

    __ascendc_ret = launch_and_profiling_calcprefix(3, blockDim, stream, (void **)&__ascendc_args, sizeof(__ascendc_args));
    KernelHandleGradUnregister::GetInstance();
    FreeAscendMemDevice(__ascendc_args.__ascendc_overflow);
    return __ascendc_ret;
}


uint32_t launch_and_profiling_coalesce(uint64_t func_key, uint32_t blockDim, void* stream, void **args, uint32_t size)
{
    uint64_t startTime;
    const char *name = "coalesce";
    bool profStatus = GetAscendProfStatus();
    if (profStatus) {
        StartAscendProf(name, &startTime);
    }
    if (g_kernel_handle_aiv == nullptr) {
        printf("[ERROR] %s\n", ascendcErrMsg);
        return 0;
    }
    uint32_t ret = LaunchAscendKernel(g_kernel_handle_aiv, func_key, blockDim, args, size, stream);
    if (ret != 0) {
        printf("LaunchAscendKernel ret %u\n", ret);
    }
    if (profStatus) {
        ReportAscendProf(name, blockDim, 1, startTime);
    }
    return ret;
}

extern "C" uint32_t aclrtlaunch_coalesce(uint32_t blockDim, void* stream, uint32_t dataBlockNum, void* finalCompressedExp, void* compressedSize, void* compressedSizePrefix, uint32_t totalUncompressedBytes)
{
    struct {
        alignas(((alignof(uint32_t) + 3) >> 2) << 2) uint32_t dataBlockNum;
        alignas(((alignof(void*) + 3) >> 2) << 2) void* finalCompressedExp;
        alignas(((alignof(void*) + 3) >> 2) << 2) void* compressedSize;
        alignas(((alignof(void*) + 3) >> 2) << 2) void* compressedSizePrefix;
        alignas(((alignof(uint32_t) + 3) >> 2) << 2) uint32_t totalUncompressedBytes;
        alignas(((alignof(void*) + 3) >> 2) << 2) void* __ascendc_overflow;
    } __ascendc_args;

    uint32_t __ascendc_ret;
    constexpr uint32_t __ascendc_overflow_status_size = 8;
    AllocAscendMemDevice(&(__ascendc_args.__ascendc_overflow), __ascendc_overflow_status_size);
    __ascendc_args.dataBlockNum = dataBlockNum;
    __ascendc_args.finalCompressedExp = finalCompressedExp;
    __ascendc_args.compressedSize = compressedSize;
    __ascendc_args.compressedSizePrefix = compressedSizePrefix;
    __ascendc_args.totalUncompressedBytes = totalUncompressedBytes;

    __ascendc_ret = launch_and_profiling_coalesce(4, blockDim, stream, (void **)&__ascendc_args, sizeof(__ascendc_args));
    KernelHandleGradUnregister::GetInstance();
    FreeAscendMemDevice(__ascendc_args.__ascendc_overflow);
    return __ascendc_ret;
}
