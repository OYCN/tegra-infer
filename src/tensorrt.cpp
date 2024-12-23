//
// Created by oPluss on 2024/12/21.
//

#include "tensorrt.h"
#include "fs.h"
#include "macro.h"

void TrtLogger::log(Severity severity, const char* msg) noexcept {
    LOG(INFO) << msg;
}

TrtInfer::TrtInfer(nvinfer1::IExecutionContext* ctx)
    :ctx_(ctx) {}


void TrtInfer::bind(const std::string& name, const void* ptr) {
    ctx_->setInputTensorAddress(name.c_str(), ptr);
}
void TrtInfer::bind(const std::string& name, void* ptr) {
    ctx_->setTensorAddress(name.c_str(), ptr);
}
void TrtInfer::check_all_bind() {
    auto& e = ctx_->getEngine();
    bool success = true;
    for (int i = 0; i < e.getNbIOTensors(); i++) {
        auto name = e.getIOTensorName(i);
        auto ptr = ctx_->getTensorAddress(name);
        if (ptr == nullptr) {
            LOG(ERROR) << "IO of TRT is not set: " << name;
            success = false;
        }
    }
    CHECK(success);
}
void TrtInfer::enableCudaGraph() {
    check_all_bind();
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));
    CHECK(ctx_->enqueueV3(stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    ctx_->enqueueV3(stream);
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph_));
    CUDA_CHECK(cudaGraphInstantiate(&exec_, graph_));
}
void TrtInfer::infer(cudaStream_t stream) {
    if (exec_ != nullptr) {
        CUDA_CHECK(cudaGraphLaunch(exec_, stream));
    } else {
        check_all_bind();
        ctx_->enqueueV3(stream);
    }
}

std::vector<std::string> TrtInfer::inputNames() {
    auto& e = ctx_->getEngine();
    std::vector<std::string> ret;
    for (int i = 0; i < e.getNbIOTensors(); i++) {
        auto name = e.getIOTensorName(i);
        auto mode = e.getTensorIOMode(name);
        auto type = e.getTensorDataType(name);
        if (mode == nvinfer1::TensorIOMode::kINPUT) {
            ret.emplace_back(name);
        }
    }
    return ret;
}
std::vector<std::string> TrtInfer::outputNames() {
    auto& e = ctx_->getEngine();
    std::vector<std::string> ret;
    for (int i = 0; i < e.getNbIOTensors(); i++) {
        auto name = e.getIOTensorName(i);
        auto mode = e.getTensorIOMode(name);
        auto type = e.getTensorDataType(name);
        if (mode == nvinfer1::TensorIOMode::kOUTPUT) {
            ret.emplace_back(name);
        }
    }
    return ret;
}
std::vector<int64_t> TrtInfer::getShape(const std::string& name) {
    std::vector<int64_t> ret;
    auto dim = ctx_->getTensorShape(name.c_str());
    for (int i = 0; i < dim.nbDims; i++) {
        ret.emplace_back(dim.d[i]);
    }
    return ret;
}
std::vector<int64_t> TrtInfer::getStride(const std::string& name) {
    std::vector<int64_t> ret;
    auto dim = ctx_->getTensorStrides(name.c_str());
    for (int i = 0; i < dim.nbDims; i++) {
        ret.emplace_back(dim.d[i]);
    }
    return ret;
}
nvinfer1::DataType TrtInfer::getDataType(const std::string& name) {
    return ctx_->getEngine().getTensorDataType(name.c_str());
}
nvinfer1::TensorFormat TrtInfer::getFormat(const std::string& name) {
    return ctx_->getEngine().getTensorFormat(name.c_str());
}

TrtModels& TrtModels::getInstance() {
    if (getUniqueObj() == nullptr) {
        init();
    }
    return *getUniqueObj();
}

void TrtModels::init() {
    if (getUniqueObj() == nullptr) {
        getUniqueObj().reset(new TrtModels);
    }
}
void TrtModels::destroy() {
    getUniqueObj().reset();
}

std::unique_ptr<TrtModels>& TrtModels::getUniqueObj() {
    static std::unique_ptr<TrtModels> obj;
    return obj;
}

TrtModels::TrtModels() {
    auto* rt = nvinfer1::createInferRuntime(logger);
    CHECK_NOTNULL(rt);
    runtime_.reset(rt);
}

TrtInfer* TrtModels::load(const std::string& path) {
    if (models_.count(path) == 0) {
        auto buff = readFileToString(path);
        auto* engine = runtime_->deserializeCudaEngine(buff.data(), buff.size());
        CHECK_NOTNULL(engine);
        models_[path].reset(engine);
    }
    auto* ctx = models_.at(path)->createExecutionContext();
    CHECK_NOTNULL(ctx);
    infers_.emplace_back(ctx);
    return &infers_.back();
}
