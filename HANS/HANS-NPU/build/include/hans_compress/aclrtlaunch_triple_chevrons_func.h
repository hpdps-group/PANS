
#ifndef HEADER_ACLRTLAUNCH_MERGEHISTOGRAM_HKERNEL_H_
#define HEADER_ACLRTLAUNCH_MERGEHISTOGRAM_HKERNEL_H_



extern "C" uint32_t aclrtlaunch_MergeHistogram(uint32_t blockDim, void* stream, void* hist_in, void* table);

inline uint32_t MergeHistogram(uint32_t blockDim, void* hold, void* stream, void* hist_in, void* table)
{
    (void)hold;
    return aclrtlaunch_MergeHistogram(blockDim, stream, hist_in, table);
}

#endif

#ifndef HEADER_ACLRTLAUNCH_EXTRACTBITS_AND_HISTOGRAM_HKERNEL_H_
#define HEADER_ACLRTLAUNCH_EXTRACTBITS_AND_HISTOGRAM_HKERNEL_H_



extern "C" uint32_t aclrtlaunch_extractbits_and_histogram(uint32_t blockDim, void* stream, uint32_t datablockNum, void* in, void* tempBuffer, void* final, void* histogramDevice, uint32_t totalUncompressedSize);

inline uint32_t extractbits_and_histogram(uint32_t blockDim, void* hold, void* stream, uint32_t datablockNum, void* in, void* tempBuffer, void* final, void* histogramDevice, uint32_t totalUncompressedSize)
{
    (void)hold;
    return aclrtlaunch_extractbits_and_histogram(blockDim, stream, datablockNum, in, tempBuffer, final, histogramDevice, totalUncompressedSize);
}

#endif

#ifndef HEADER_ACLRTLAUNCH_COMP_HKERNEL_H_
#define HEADER_ACLRTLAUNCH_COMP_HKERNEL_H_



extern "C" uint32_t aclrtlaunch_comp(uint32_t blockDim, void* stream, uint32_t datablockNum, void* tempBuffer, void* final, void* histogramDevice, void* compressedSize, uint32_t totalUncompressedBytes);

inline uint32_t comp(uint32_t blockDim, void* hold, void* stream, uint32_t datablockNum, void* tempBuffer, void* final, void* histogramDevice, void* compressedSize, uint32_t totalUncompressedBytes)
{
    (void)hold;
    return aclrtlaunch_comp(blockDim, stream, datablockNum, tempBuffer, final, histogramDevice, compressedSize, totalUncompressedBytes);
}

#endif

#ifndef HEADER_ACLRTLAUNCH_CALCPREFIX_HKERNEL_H_
#define HEADER_ACLRTLAUNCH_CALCPREFIX_HKERNEL_H_



extern "C" uint32_t aclrtlaunch_calcprefix(uint32_t blockDim, void* stream, uint32_t datablockNum, void* tilePrefix, void* compressedSize, void* compressedSizePrefix);

inline uint32_t calcprefix(uint32_t blockDim, void* hold, void* stream, uint32_t datablockNum, void* tilePrefix, void* compressedSize, void* compressedSizePrefix)
{
    (void)hold;
    return aclrtlaunch_calcprefix(blockDim, stream, datablockNum, tilePrefix, compressedSize, compressedSizePrefix);
}

#endif

#ifndef HEADER_ACLRTLAUNCH_COALESCE_HKERNEL_H_
#define HEADER_ACLRTLAUNCH_COALESCE_HKERNEL_H_



extern "C" uint32_t aclrtlaunch_coalesce(uint32_t blockDim, void* stream, uint32_t dataBlockNum, void* finalCompressedExp, void* compressedSize, void* compressedSizePrefix, uint32_t totalUncompressedBytes);

inline uint32_t coalesce(uint32_t blockDim, void* hold, void* stream, uint32_t dataBlockNum, void* finalCompressedExp, void* compressedSize, void* compressedSizePrefix, uint32_t totalUncompressedBytes)
{
    (void)hold;
    return aclrtlaunch_coalesce(blockDim, stream, dataBlockNum, finalCompressedExp, compressedSize, compressedSizePrefix, totalUncompressedBytes);
}

#endif
