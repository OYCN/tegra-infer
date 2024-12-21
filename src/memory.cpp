//
// Created by oPluss on 2024/12/21.
//

#include "memory.h"
#include "macro.h"

CudaMemory::CudaMemory(size_t size) : IMemory(size, MemoryType::kCuda) {
    if (size > 0) {
        CUDA_CHECK(cudaMalloc(&ptr_, size));
        CHECK_NOTNULL(ptr_);
    } else {
        ptr_ = nullptr;
    }
}

CudaMemory::~CudaMemory() {
    CUDA_CHECK(cudaFree(ptr_));
}

MappedMemory::MappedMemory(size_t size) : IMemory(size, MemoryType::kMapped) {
    if (size > 0) {
        CUDA_CHECK(cudaMallocHost(&ptr_, size));
        CHECK_NOTNULL(ptr_);
        void* tmp = nullptr;
        CUDA_CHECK(cudaHostGetDevicePointer(&tmp, ptr_, 0));
        CHECK_EQ(tmp, ptr_);
    } else {
        ptr_ = nullptr;
    }
}

MappedMemory::~MappedMemory() {
    CUDA_CHECK(cudaFreeHost(ptr_));
}
