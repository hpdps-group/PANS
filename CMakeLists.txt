
cmake_minimum_required(VERSION 3.15)
project(CPUANS LANGUAGES CXX)

find_package(OpenMP REQUIRED)
if(OpenMP_FOUND)
  add_compile_options(${OpenMP_CXX_FLAGS})
  link_libraries(${OpenMP_CXX_FLAGS})
endif()

add_executable(cpuans_compress
  compress.cpp
)

target_link_libraries(cpuans_compress PRIVATE
  OpenMP::OpenMP_CXX
)

add_executable(cpuans_decompress
  decompress.cpp
)

target_link_libraries(cpuans_decompress PRIVATE
  OpenMP::OpenMP_CXX
)

if(APPLE)
  find_path(LLVM_DIR "llvm" PATHS "/opt/homebrew/opt/llvm" REQUIRED)
  include_directories(${LLVM_DIR}/include)
  link_directories(${LLVM_DIR}/lib)
endif()