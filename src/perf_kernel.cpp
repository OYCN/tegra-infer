//
// Created by nv on 24-12-23.
//

#include "kernel.h"
#include "memory.h"
#include "macro.h"

int main() {
    CudaMemory netout(84 * 8400 * sizeof(float));
    netout.LoadFrom("netout.dat");
    CudaMemory nb_d(sizeof(int32_t));
    MappedMemory nb_h(sizeof(int32_t));
    MappedMemory output(8400 * sizeof(Box));
    DecoderParam param = {
        .tensor = reinterpret_cast<float*>(netout.GetPtr()),
        .valid_nb_ptr = reinterpret_cast<int32_t*>(nb_d.GetPtr()),
        .boxes = reinterpret_cast<Box*>(output.GetPtr()),
        .pad_l = 80,
        .pad_t = 0,
        .scale = 1.6,
        .len = 8400,
        .cls = 84,
        .score_thr = 0.25
    };
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    for (int i = 0; i < 100; i++) {
        decoder(param, nullptr);
    }
    CUDA_CHECK(cudaEventRecord(start, nullptr));
    for (int i = 0; i < 100; i++) {
        decoder(param, nullptr);
    }
    CUDA_CHECK(cudaEventRecord(stop, nullptr));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = -1;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    LOG(INFO) << "time: " << ms / 100 << " ms";
    return 0;
}
