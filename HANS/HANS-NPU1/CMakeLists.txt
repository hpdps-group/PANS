# Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.

# CMake lowest version requirement
cmake_minimum_required(VERSION 3.16.0)

# project information
project(Ascend_C)
set(SOC_VERSION "Ascend910B2" CACHE STRING "system on chip type")
# add_ops_compile_options(ALL OPTIONS -DASCENDC_DUMP=0)

if(DEFINED ENV{USER} AND "$ENV{USER}" STREQUAL "root")
    set(DEFAULT_ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/ascend-toolkit/latest" CACHE PATH "ASCEND CANN package default installation directory for root user")
else()
    set(DEFAULT_ASCEND_CANN_PACKAGE_PATH "$ENV{HOME}/Ascend/ascend-toolkit/latest" CACHE PATH "ASCEND CANN package default installation directory for other user")
endif()

if(DEFINED ASCEND_CANN_PACKAGE_PATH)
elseif(DEFINED ENV{ASCEND_HOME_PATH})
    set(ASCEND_CANN_PACKAGE_PATH "$ENV{ASCEND_HOME_PATH}" CACHE PATH "ASCEND CANN package installation directory" FORCE)
else()
    set(ASCEND_CANN_PACKAGE_PATH "${DEFAULT_ASCEND_CANN_PACKAGE_PATH}" CACHE PATH "ASCEND CANN package installation directory")
endif()

set(RUN_MODE "npu" CACHE STRING "run mode: npu")
set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type Release/Debug (default Debug)" FORCE)
set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_LIST_DIR}/out" CACHE STRING "path for install()" FORCE)

if(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/ascendc_devkit/tikcpp/samples/cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/ascendc_devkit/tikcpp/samples/cmake)
else()
    message(FATAL_ERROR "ascendc_kernel_cmake does not exist, please check whether the cann package is installed.")
endif()

include(${ASCENDC_CMAKE_DIR}/ascendc.cmake)
# set(Torch_DIR "/path/to/libtorch/share/cmake/Torch")
# find_package(Torch_npu REQUIRED)
# find_package(Torch REQUIRED)
# set(Torch_npu "/root/yjw/pytorch/libtorch_npu/")
# set(Torch_DIR "/root/yjw/libtorch/share/cmake/Torch")  # 关键路径
# find_package(Torch REQUIRED)
# set(torch_npu_path "/root/yjw/pytorch/libtorch_npu/")         # 设置libtorch_npu的路径
# include_directories(${torch_npu_path}/include)   # 设置引用libtorch_npu的头文件路径
# link_directories(${torch_npu_path}/lib)          # 设置引用libtorch_npu的库文件路径


# ascendc_library use to add kernel file to generate ascendc library
ascendc_library(hans_compress STATIC
    hans_table.cpp
    hans_compress.cpp
    hans_merge.cpp
)
# ascendc_library(hans_decompress STATIC
#     hans_decompress.cpp
# )

add_executable(compress compress.cpp)
# add_executable(decompress decompress.cpp)

target_link_libraries(compress PRIVATE
    hans_compress
    # "${TORCH_LIBRARIES}"
    # torch_npu
    # "${TORCH_NPU_LIBRARIES}"
)
# target_link_libraries(decompress PRAVITE
#     hans_decompress
# )
