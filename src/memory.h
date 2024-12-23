//
// Created by oPluss on 2024/12/21.
//

#ifndef MEMORY_H
#define MEMORY_H

#include <cstddef>
#include <string>

#define DELETE_COPY_AND_ASSIGN(cls) \
cls(const cls&) = delete; \
cls operator=(const cls&) = delete;

enum class MemoryType {
  kNone = 0,
  kPageable,
  kPagelock,
  kCuda,
  kMapped
};

class IMemory {
  public:
    IMemory(size_t size, MemoryType type) : size_(size), type_(type) {}
    virtual ~IMemory() = default;

    MemoryType GetMemoryType() const { return type_; }
    void* GetPtr() const { return ptr_; }
    size_t GetSize() const { return size_; }

    void LoadFrom(const std::string& path);
    void SaveTo(const std::string& path);

  protected:
    void* ptr_ = nullptr;
    size_t size_ = 0;
    MemoryType type_ = MemoryType::kNone;
};

class CudaMemory : public IMemory {
  public:
    explicit CudaMemory(size_t size);
    ~CudaMemory() final;

  DELETE_COPY_AND_ASSIGN(CudaMemory);
};

class MappedMemory : public IMemory {
  public:
    explicit MappedMemory(size_t size);
    ~MappedMemory() final;

  DELETE_COPY_AND_ASSIGN(MappedMemory);
};

#endif //MEMORY_H
