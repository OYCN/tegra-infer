//
// Created by nv on 24-12-23.
//

#include "source.h"
#include <nvtx3/nvToolsExt.h>

CameraSource::CameraSource(const nlohmann::json& j) {
    auto uri = j["device"].get<std::string>();
    auto w = j["raw_w"].get<uint64_t>();
    auto h = j["raw_h"].get<uint64_t>();
    auto fps = j["raw_fps"].get<float>();
    auto buffer_nb = j["buffer_nb"].get<uint64_t>();

    LOG(INFO) << "camera";
    LOG(INFO) << "  uri      : " << uri;
    LOG(INFO) << "  w        : " << w;
    LOG(INFO) << "  h        : " << h;
    LOG(INFO) << "  fps      : " << fps;
    LOG(INFO) << "  buffer_nb: " << buffer_nb;

    videoOptions opt;
    opt.resource = uri.c_str();
    opt.width = w;
    opt.height = h;
    opt.frameRate = fps;
    opt.numBuffers = buffer_nb;
    opt.zeroCopy = false;
    opt.flipMethod = videoOptions::FlipMethod::FLIP_ROTATE_180;
    camera_.reset(gstCamera::Create(opt));
}
int CameraSource::getH() {
    return camera_->GetHeight();
}
int CameraSource::getW() {
    return camera_->GetWidth();
}
const std::vector<Frame>& CameraSource::getFrames(cudaStream_t stream) {
    nvtxRangePushA("Pipline::process::camera");
    frames_.resize(1);
    auto& frame = frames_.back();
    int status = 0;

    CHECK(camera_->Capture(&frame.ptr, imageFormat::IMAGE_RGB32F, gstCamera::DEFAULT_TIMEOUT, &status, stream)) << "status: " << status;
    frame.format = imageFormat::IMAGE_RGB32F;

    frame.ts = camera_->GetLastTimestamp();
    frame.h = camera_->GetHeight();
    frame.w = camera_->GetWidth();
    nvtxRangePop();
    return frames_;
}

ImageSource::ImageSource(const nlohmann::json& j) {
    path_ = j["path"].get<std::string>();
    void* ptr = nullptr;
    int w = 0;
    int h = 0;
    CHECK(loadImage(path_.c_str(), &ptr, &w, &h, imageFormat::IMAGE_RGB32F, nullptr));
    CUDA_CHECK(cudaStreamSynchronize(nullptr));
    frames_ = {
        Frame{
            .ts = 0,
            .w = static_cast<uint64_t>(w),
            .h = static_cast<uint64_t>(h),
            .ptr = ptr,
            .format = imageFormat::IMAGE_RGB32F
        }
    };
}
int ImageSource::getH() {
    return frames_.at(0).h;
}
int ImageSource::getW() {
    return frames_.at(0).w;
}
const std::vector<Frame>& ImageSource::getFrames(cudaStream_t stream) {
    void* ptr = nullptr;
    int w = 0;
    int h = 0;
    CHECK(loadImage(path_.c_str(), &ptr, &w, &h, imageFormat::IMAGE_RGB32F, stream));
    frames_.at(0).ptr = ptr;
    return frames_;
}
