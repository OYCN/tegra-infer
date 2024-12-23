//
// Created by nv on 24-12-22.
//

#ifndef TYPEDEF_H
#define TYPEDEF_H

#include <vector>
#include <cuda_fp16.h>
#include "3rd/utils/image/imageFormat.h"

struct half4 {
    half x, y, z, w;
};

inline __device__ __host__ float2 operator*(const float2& a, const float2& b) {
    return make_float2(a.x * b.x, a.y * b.y);
}
inline __device__ __host__ float2 operator+(const float2& a, const float2& b) {
    return make_float2(a.x + b.x, a.y + b.y);
}
inline __device__ __host__ float3 operator*(const float3& a, const float3& b) {
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}
inline __device__ __host__ float3 operator+(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline __device__ __host__ float3 operator-(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
inline __device__ __host__ float3 operator*(float a, const float3 b) {
    return make_float3(a * b.x, a * b.y, a * b.z);
}

template<typename TA, typename TB>
struct TypeCvt;

template<>
struct TypeCvt<uchar4, float3> {
    static inline __device__ __host__ float3 call(const uchar4& a) {
        return make_float3(
            static_cast<float>(a.x),
            static_cast<float>(a.y),
            static_cast<float>(a.z)
        );
    }
};
template<>
struct TypeCvt<float3, half4> {
    static inline __device__ __host__ half4 call(const float3& a) {
        return {a.x, a.y, a.z, 0};
    }
};
template<>
struct TypeCvt<float3, float3> {
    static inline __device__ __host__ float3 call(const float3& a) {
        return a;
    }
};

struct Frame {
    uint64_t ts = 0;
    uint64_t w = 0;
    uint64_t h = 0;
    void* ptr = nullptr;
    imageFormat format = imageFormat::IMAGE_UNKNOWN;
};

struct Box {
    float x0;
    float y0;
    float x1;
    float y1;
    int type;
    float score;
};

struct DecoderParam {
    const float* tensor;
    int32_t* valid_nb_ptr;
    Box* boxes;
    int pad_l;
    int pad_t;
    float scale;
    int len;
    int cls;
    float score_thr = 0.25;
};

struct Result {
    std::vector<Box> boxes;
};

#endif //TYPEDEF_H
