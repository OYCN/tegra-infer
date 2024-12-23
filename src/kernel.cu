//
// Created by nv on 24-12-22.
//

#include "kernel.h"

#include "macro.h"

__global__ void decoder_kernel(DecoderParam param) {
    for (int tid = blockIdx.x * blockDim.x + threadIdx.x;
         tid < param.len;
         tid += blockDim.x * gridDim.x) {
        int cls = -1;
        float score = -1.f;
        for (int c = 4; c < param.cls; c++) {
            float s = param.tensor[c * param.len + tid];
            if (s > score) {
                cls = c;
                score = s;
            }
        }
        if (score > param.score_thr) {
            // printf("tid: %d, score: %f, cls: %d\n", tid, score, cls);
            auto id = atomicAdd(param.valid_nb_ptr, 1);
            float x = param.tensor[0 * param.len + tid] - param.pad_l;
            float y = param.tensor[1 * param.len + tid] - param.pad_t;
            float w_2 = param.tensor[2 * param.len + tid] * 0.5f;
            float h_2 = param.tensor[3 * param.len + tid] * 0.5f;
            param.boxes[id] = {
                .x0 = (x - w_2) * param.scale,
                .y0 = (y - h_2) * param.scale,
                .x1 = (x + w_2) * param.scale,
                .y1 = (y + h_2) * param.scale,
                .type = cls,
                .score = score
            };
        }
    }
}

void decoder(const DecoderParam& param, cudaStream_t stream) {
    CUDA_CHECK(cudaMemsetAsync(param.valid_nb_ptr, 0, sizeof(int32_t), stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    auto block = dim3(512, 1, 1);
    auto grid = dim3(32, 1, 1);
    decoder_kernel<<<grid, block, 0, stream>>>(param);
    CUDA_CHECK(cudaStreamSynchronize(stream));
}
