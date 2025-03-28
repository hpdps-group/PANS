# PANS

(C) 2025 by Institute of Computing Technology, Chinese Academy of Sciences. 
- Developer: Jinwu Yang 
- Advisor: Dingwen Tao, Guangming Tan

PANS is a parallel implementation of the Asymmetric Numeral Systems (ANS) compression algorithm. It supports asymmetric compression and decompression across different hardware architectures, enabling flexible deployment based on system requirements. Specifically, it allows:
- Parallel compression on NVIDIA/AMD GPUs using [dietGPU](https://github.com/facebookresearch/dietgpu/), with parallel decompression on multi-core CPUs.
- Parallel compression on multi-core CPUs using PANS, with parallel decompression on NVIDIA GPUs via [dietGPU](https://github.com/facebookresearch/dietgpu/).

## Building

Clone this repo using

```shell
git clone https://github.com/hpdps-group/PANS.git
```

Do the standard CMake thing:

```shell
cd PANS; mkdir build; cd build;
cmake .. && make
```
## Run

```shell
compress: ./cpuans_compress input_file temp_file
decompress: ./cpuans_decompress temp_file output_file
```
