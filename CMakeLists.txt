cmake_minimum_required(VERSION 3.20)

set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")

project(tegra_infer LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 20)

find_package(CUDAToolkit)

add_subdirectory(3rd)

add_library(tegra_infer SHARED library.cpp)

add_subdirectory(tools)
