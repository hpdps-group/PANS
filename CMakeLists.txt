# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.
#set(CMAKE_MODULE_PATH "/usr/local/python3/lib/python3.8/site-packages/torch/cmake/modules" ${CMAKE_MODULE_PATH})

cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_COMPILER /opt/dtk-25.04/bin/hipcc)
set(CMAKE_CXX_LINKER /opt/dtk-25.04/bin/hipcc)
project(hipANS LANGUAGES HIP CXX VERSION 1.0)

set(CMAKE_CXX_STANDARD 14)
if(NOT DEFINED HIP_PATH)
if(NOT DEFINED ENV{HIP_PATH})
set(HIP_PATH "/opt/rocm/hip" CACHE PATH "Path to which HIP has been installed")
else()
set(HIP_PATH $ENV{HIP_PATH} CACHE PATH "Path to which HIP has been installed")
endif()
endif()
set(CMAKE_MODULE_PATH "${HIP_PATH}/cmake" ${CMAKE_MODULE_PATH})
find_package(HIP)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

#hip_add_library
add_library
(gpu_ans SHARED
  src/GpuANS.cpp
)

target_include_directories(gpu_ans PUBLIC
   $<BUILD_INTERFACE:${hipANS_SOURCE_DIR}>
  "${TORCH_INCLUDE_DIRS}"
)
target_link_libraries(gpu_ans PUBLIC
  "${TORCH_LIBRARIES}"
)

target_include_directories(gpu_ans
      PUBLIC
      # where the library itself will look for its internal headers
      ${CMAKE_CURRENT_SOURCE_DIR}/src
      PUBLIC
      # where top-level project will look for the library's public headers
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
      # where external projects will look for the library's public headers
      $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# hip_add_executable
add_library
(hipans_compress examples/compress.cpp)
target_link_libraries(hipans_compress
        gpu_ans)

# hip_add_executable
add_library
(hipans_decompress examples/decompress.cpp)
target_link_libraries(hipans_decompress
        gpu_ans)
        