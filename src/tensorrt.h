//
// Created by oPluss on 2024/12/21.
//

#ifndef TENSORRT_H
#define TENSORRT_H

#include <map>
#include <memory>
#include <string>
#include "glog/logging.h"
#include "NvInfer.h"

class TrtLogger: public nvinfer1::ILogger
{
public:
    void log(Severity severity, const char* msg) noexcept override;
};

class TrtInfer {
public:
    explicit TrtInfer(nvinfer1::IExecutionContext* ctx);

    void bind(const std::string& name, const void* ptr);
    void bind(const std::string& name, void* ptr);
    void enableCudaGraph();
    void infer(cudaStream_t stream);

    std::vector<std::string> inputNames();
    std::vector<std::string> outputNames();
    std::vector<int64_t> getShape(const std::string& name);
    std::vector<int64_t> getStride(const std::string& name);
    nvinfer1::DataType getDataType(const std::string& name);
    nvinfer1::TensorFormat getFormat(const std::string& name);

private:
    void check_all_bind();

    std::unique_ptr<nvinfer1::IExecutionContext> ctx_;
    cudaGraph_t graph_;
    cudaGraphExec_t exec_;
};

class TrtModels {
public:
    static TrtModels& getInstance();
    static void init();
    static void destroy();

    TrtInfer* load(const std::string& path);

private:
    TrtModels();
    static std::unique_ptr<TrtModels>& getUniqueObj();
    TrtLogger logger;
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::map<std::string, std::unique_ptr<nvinfer1::ICudaEngine>> models_;
    std::vector<TrtInfer> infers_;
};

#endif //TENSORRT_H
