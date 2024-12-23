//
// Created by nv on 24-12-21.
//

#ifndef STAGE_H
#define STAGE_H

#include <vector>
#include "macro.h"
#include "tensorrt.h"
#include "typedef.h"
#include "memory.h"

#include "3rd/nlohmann/json.hpp"
#include "3rd/utils/cuda/cudaFont.h"

class Stage {
public:
    Stage() = default;
    void init(const nlohmann::json& j);
    void gpuProcess(const Frame& img, cudaStream_t stream);
    void parseResults();
    Frame genRenderImage(cudaStream_t stream, bool mock = false);
    const Result& getResult() const {return result_;}
private:
    TrtInfer* infer_;
    Frame raw_img_;

    Frame resized_img_;
    std::unique_ptr<IMemory> resized_image_buffer_;

    Frame netin_img_;
    std::unique_ptr<IMemory> netin_buffer_;
    std::unique_ptr<IMemory> netout_buffer_;

    DecoderParam decoder_param;
    std::unique_ptr<IMemory> decoder_box_nb_d_buffer_;
    std::unique_ptr<IMemory> decoder_box_nb_h_buffer_;
    std::unique_ptr<IMemory> decoder_boxes_buffer_;

    uint32_t netin_h_ = 0;
    uint32_t netin_w_ = 0;
    float scale_ = 1;
    uint32_t pad_t = 0;
    uint32_t pad_l = 0;
    float nms_thr_ = 0;
    Result result_;

    std::vector<std::string> type_map_;
    std::unique_ptr<cudaFont> font_;
};

#endif //STAGE_H
