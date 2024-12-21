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

#endif //MACRO_H
