//
// Created by oPluss on 2024/12/21.
//

#ifndef MACRO_H
#define MACRO_H

#include "cuda_runtime_api.h"
#include "glog/logging.h"

#define CUDA_CHECK(call) \
  {\
  auto ret = (call); \
    if (ret != cudaSuccess) { \
    LOG(FATAL) << "CUDA error (" << #call << "): " << cudaGetErrorString(ret); \
    }\
  }

#define CUDA_LOOP_2D(tx, ty, w, h) \
    for (int ty = blockIdx.y * blockDim.y + threadIdx.y; ty < h; ty += blockDim.y * gridDim.y)\
    for (int tx = blockIdx.x * blockDim.x + threadIdx.x; tx < w; tx += blockDim.x * gridDim.x)

#endif //MACRO_H
