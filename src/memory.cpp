//
// Created by oPluss on 2024/12/21.
//

#include "memory.h"
#include "macro.h"
#include "fs.h"

void IMemory::LoadFrom(const std::string& path) {
    auto buff = readFileToString(path);
    CHECK_EQ(buff.size(), GetSize());
    CUDA_CHECK(cudaMemcpy(GetPtr(), buff.data(), GetSize(), cudaMemcpyDefault));
    CUDA_CHECK(cudaStreamSynchronize(nullptr));
}

void IMemory::SaveTo(const std::string& path) {
    std::string buff;
    buff.resize(GetSize());
    CUDA_CHECK(cudaMemcpy(buff.data(), GetPtr(), GetSize(), cudaMemcpyDefault));
    CUDA_CHECK(cudaStreamSynchronize(nullptr));
    writeStringToFile(path, buff);
}

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
