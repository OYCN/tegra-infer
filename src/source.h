//
// Created by nv on 24-12-23.
//

#ifndef SOURCE_H
#define SOURCE_H

#include "macro.h"
#include "typedef.h"
#include "memory.h"
#include <memory>
#include "3rd/utils/camera/gstCamera.h"
#include "3rd/utils/image/loadImage.h"
#include "3rd/nlohmann/json.hpp"

class ISource {
public:
    virtual ~ISource() = default;

    virtual int getH() = 0;
    virtual int getW() = 0;
    virtual const std::vector<Frame>& getFrames(cudaStream_t stream) = 0;
};

class CameraSource: public ISource {
public:
    CameraSource(const nlohmann::json& j);
    int getH() override;
    int getW() override;
    const std::vector<Frame>& getFrames(cudaStream_t stream) override;

private:
    std::unique_ptr<gstCamera> camera_;
    std::vector<Frame> frames_;
};

class ImageSource: public ISource {
public:
    ImageSource(const nlohmann::json& j);
    int getH() override;
    int getW() override;
    const std::vector<Frame>& getFrames(cudaStream_t stream) override;

private:
    std::string path_;
    std::vector<Frame> frames_;
};

#endif //SOURCE_H
