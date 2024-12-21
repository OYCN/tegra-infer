cmake_minimum_required(VERSION 3.20)

set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")

project(tegra_infer LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 20)

find_package(CUDAToolkit)

add_subdirectory(3rd)
add_subdirectory(src)
add_subdirectory(tools)
