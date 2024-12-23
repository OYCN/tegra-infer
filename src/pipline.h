//
// Created by nv on 24-12-21.
//

#ifndef PIPLINE_H
#define PIPLINE_H


#include <vector>
#include "stage.h"
#include "macro.h"
#include "typedef.h"
#include "source.h"
#include <thread>
#include "3rd/utils/display/glDisplay.h"

class RingUint64 {
public:
    RingUint64(size_t val, size_t max_val):
    val_(val), max_val_(max_val) {
    }
    inline size_t inc_() {
        return (++val_)%max_val_;
    }
    inline size_t inc() {
        return (val_ + 1)%max_val_;
    }
    inline size_t dec() {
        if (val_ == 0) {
            return max_val_-1;
        }
        return val_ - 1;
    }
    size_t val() const {return val_;}
private:
    size_t val_;
    size_t min_val_;
    size_t max_val_;
};


class Pipline {
public:
    Pipline(const nlohmann::json& j);

    void process();

private:
    std::vector<Frame> getFrames(cudaStream_t stream);
    void display(const Frame& frame, cudaStream_t stream);

    std::vector<std::unique_ptr<ISource>> sources_;

    std::unique_ptr<glDisplay> display_;

    cudaStream_t stream_;
    Stage stage_;

    bool quiet_;

    cudaEvent_t camera_start_;
    cudaEvent_t camera_stop_;
    cudaEvent_t stage_process_start_;
    cudaEvent_t stage_process_stop_;
    cudaEvent_t stage_render_start_;
    cudaEvent_t stage_render_stop_;
    cudaEvent_t display_start_;
    cudaEvent_t display_stop_;
};

#endif //PIPLINE_H
