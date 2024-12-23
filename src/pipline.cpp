//
// Created by nv on 24-12-21.
//

#include "pipline.h"

#include <chrono>

#include "nvtx3/nvToolsExt.h"

#include "3rd/utils/video/videoOptions.h"
#include "3rd/utils/image/imageFormat.h"

Pipline::Pipline(const nlohmann::json& j) {
    // config
    quiet_ = j["quiet"].get<bool>();

    // source
    auto sources = j["sources"].get<std::vector<std::string>>();
    int max_w = 0;
    int max_h = 0;
    for (const auto& source : sources) {
        CHECK_EQ(j.count(source), 1);
        const auto& cfg = j[source];
        if (source == "camera") {
            sources_.emplace_back(new CameraSource(cfg));
        } else if (source == "image") {
            sources_.emplace_back(new ImageSource(cfg));
        } else {
            LOG(FATAL) << "unsupport source: " << source;
        }
        auto w = sources_.back()->getW();
        auto h = sources_.back()->getH();
        if (w < 0 || h < 0) {
            max_w = -1;
            max_h = -1;
        }
        if (max_h >=0 && max_w >= 0) {
            max_w = std::max(max_w, w);
            max_h = std::max(max_h, h);
        }
    }

    // stage
    stage_.init(j);

    // display
    if (j["display"].get<bool>()) {
        LOG(INFO) << "display w: " << max_w;
        LOG(INFO) << "display h: " << max_h;
        display_.reset(glDisplay::Create("display", max_w, max_h));
    }

    // other
    CUDA_CHECK(cudaStreamCreate(&stream_));
    CUDA_CHECK(cudaEventCreate(&camera_start_));
    CUDA_CHECK(cudaEventCreate(&camera_stop_));
    CUDA_CHECK(cudaEventCreate(&stage_process_start_));
    CUDA_CHECK(cudaEventCreate(&stage_process_stop_));
    CUDA_CHECK(cudaEventCreate(&stage_render_start_));
    CUDA_CHECK(cudaEventCreate(&stage_render_stop_));
    CUDA_CHECK(cudaEventCreate(&display_start_));
    CUDA_CHECK(cudaEventCreate(&display_stop_));
}

void Pipline::process() {
    // nvtxRangePushA("Pipline::process");
    if (!quiet_) {
        LOG(INFO) << "=== Pipline::process start ===";
    }
    auto all_start = std::chrono::high_resolution_clock::now();

    CUDA_CHECK(cudaEventRecord(camera_start_, stream_));
    auto frames = getFrames(stream_);
    CUDA_CHECK(cudaEventRecord(camera_stop_, stream_));

    CHECK_EQ(frames.size(), 1);

    CUDA_CHECK(cudaEventRecord(stage_process_start_, stream_));
    stage_.gpuProcess(frames.at(0), stream_);
    CUDA_CHECK(cudaEventRecord(stage_process_stop_, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    auto stage_reslut_start = std::chrono::high_resolution_clock::now();
    stage_.parseResults();
    auto stage_reslut_stop = std::chrono::high_resolution_clock::now();

    CUDA_CHECK(cudaEventRecord(stage_render_start_, stream_));
    Frame display_frame;
    if (display_) {
        display_frame = stage_.genRenderImage(stream_);
    }
    CUDA_CHECK(cudaEventRecord(stage_render_stop_, stream_));

    CUDA_CHECK(cudaEventRecord(display_start_, stream_));
    if (display_) {
        display(display_frame, stream_);
    }
    CUDA_CHECK(cudaEventRecord(display_stop_, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    auto all_stop = std::chrono::high_resolution_clock::now();

    if (!quiet_) {
        {
            LOG(INFO) << "final det nb: " << stage_.getResult().boxes.size();
        }
        {
            float ms = 0;
            CUDA_CHECK(cudaEventElapsedTime(&ms, camera_start_, camera_stop_));
            LOG(INFO) << "[GPU] Pipline/camera: " << ms << " ms";
        }
        {
            float ms = 0;
            CUDA_CHECK(cudaEventElapsedTime(&ms, stage_process_start_, stage_process_stop_));
            LOG(INFO) << "[GPU] Pipline/stage/gpu_process: " << ms << " ms";
        }
        {
            auto ms = std::chrono::duration_cast<std::chrono::microseconds>(stage_reslut_stop - stage_reslut_start).count() / 1000;
            LOG(INFO) << "[CPU] Pipline/stage/parse_result: " << ms << " ms";
        }
        {
            float ms = 0;
            CUDA_CHECK(cudaEventElapsedTime(&ms, stage_render_start_, stage_render_stop_));
            LOG(INFO) << "[GPU] Pipline/stage/render: " << ms << " ms";
        }
        {
            float ms = 0;
            CUDA_CHECK(cudaEventElapsedTime(&ms, display_start_, display_stop_));
            LOG(INFO) << "[GPU] Pipline/display: " << ms << " ms";
        }
        {
            auto ms = std::chrono::duration_cast<std::chrono::microseconds>(all_stop - all_start).count() / 1000;
            LOG(INFO) << "[CPU] Pipline: " << ms << " ms";
        }
        LOG(INFO) << "=== Pipline::process done ===";
    }
    // nvtxRangePop();
}

std::vector<Frame> Pipline::getFrames(cudaStream_t stream) {
    std::vector<Frame> frames;
    for (auto& source : sources_) {
        const auto& fs = source->getFrames(stream);
        frames.insert(frames.end(), fs.begin(), fs.end());
    }
    return frames;
}

void Pipline::display(const Frame& frame, cudaStream_t stream) {
    nvtxRangePushA("Pipline::process::render");
    display_->RenderOnce(frame.ptr, frame.w, frame.h, frame.format, 5.0f, 30.0f, true, stream);
    nvtxRangePop();
}
