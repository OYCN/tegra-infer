//
// Created by nv on 24-12-21.
//

#include "stage.h"

#include "fs.h"
#include "kernel.h"

#include "3rd/utils/cuda/cudaColorspace.h"
#include "3rd/utils/cuda/cudaResize.h"
#include "3rd/utils/cuda/cudaNormalize.h"
#include "3rd/utils/cuda/cudaDraw.h"
#include "3rd/utils/cuda/cudaOverlay.h"

#include "nvtx3/nvToolsExt.h"

void Stage::init(const nlohmann::json& j) {
    auto engine_path = j["trt_engine"].get<std::string>();
    infer_ = TrtModels::getInstance().load(engine_path);

    auto inputs = infer_->inputNames();
    auto outputs = infer_->outputNames();
    CHECK_EQ(inputs.size(), 1);
    CHECK_EQ(outputs.size(), 1);
    auto input_shape = infer_->getShape(inputs.at(0));
    CHECK_EQ(input_shape.size(), 4);
    CHECK_EQ(input_shape.at(0), 1);
    CHECK_EQ(input_shape.at(1), 3);
    netin_h_ = input_shape.at(2);
    netin_w_ = input_shape.at(3);
    CHECK_GE(netin_h_, 0);
    CHECK_GE(netin_w_, 0);
    auto input_type = infer_->getDataType(inputs.at(0));
    CHECK(input_type == nvinfer1::DataType::kFLOAT);
    auto input_format = infer_->getFormat(inputs.at(0));
    CHECK(input_format == nvinfer1::TensorFormat::kHWC);

    /**
     * img -> (resize) -> resized_img_/resized_image_buffer_
     * (fill) -> netin_img_/netin_buffer_
     * resized_img_/resized_image_buffer_ -> (overlay) -> netin_img_/netin_buffer_
     */

    resized_image_buffer_ = std::make_unique<CudaMemory>(netin_w_ * netin_h_ * 3 * sizeof(float));
    resized_img_.ptr = resized_image_buffer_->GetPtr();
    resized_img_.format = imageFormat::IMAGE_RGB32F;
    resized_img_.w = 0;
    resized_img_.h = 0;
    resized_img_.ts = 0;

    netin_buffer_ = std::make_unique<CudaMemory>(netin_w_ * netin_h_ * 3 * sizeof(float));
    netin_img_.ptr = netin_buffer_->GetPtr();
    netin_img_.format = imageFormat::IMAGE_RGB32F;
    netin_img_.w = netin_w_;
    netin_img_.h = netin_h_;
    netin_img_.ts = 0;

    infer_->bind(inputs.at(0), netin_img_.ptr);

    auto output_shape = infer_->getShape(outputs.at(0));
    auto output_stride = infer_->getStride(outputs.at(0));
    size_t bytes = sizeof(float);
    for (auto v : output_shape) {
        bytes *= v;
    }
    netout_buffer_ = std::make_unique<MappedMemory>(bytes);
    infer_->bind(outputs.at(0), netout_buffer_->GetPtr());
    infer_->enableCudaGraph();

    CHECK_EQ(output_shape.size(), 3);
    CHECK_EQ(output_shape.at(0), 1);
    auto c = output_shape.at(1);
    auto len = output_shape.at(2);
    CHECK_GT(c, 4);
    CHECK_GT(len, 0);
    decoder_param.cls = c;
    decoder_param.len = len;

    decoder_box_nb_d_buffer_ = std::make_unique<CudaMemory>(sizeof(int32_t));
    decoder_box_nb_h_buffer_ = std::make_unique<MappedMemory>(sizeof(int32_t));
    decoder_boxes_buffer_ = std::make_unique<MappedMemory>(len * sizeof(Box));
    decoder_param.tensor = reinterpret_cast<float*>(netout_buffer_->GetPtr());
    decoder_param.valid_nb_ptr = reinterpret_cast<int32_t*>(decoder_box_nb_d_buffer_->GetPtr());
    decoder_param.boxes = reinterpret_cast<Box*>(decoder_boxes_buffer_->GetPtr());
}

void Stage::gpuProcess(const Frame& img, cudaStream_t stream) {
    nvtxRangePushA("Stage::gpuProcess");
    raw_img_ = img;

    // std::string buff;
    // buff.resize(img.w * img.h * 3 * sizeof(float));
    // CUDA_CHECK(cudaMemcpy(buff.data(), raw_img_.ptr, buff.size(), cudaMemcpyDefault));
    // CUDA_CHECK(cudaStreamSynchronize(nullptr));
    // writeStringToFile("raw_img_buff.dat", buff);

    float scale_h = float(img.h) / netin_h_;
    float scale_w = float(img.w) / netin_w_;
    float max_scale = std::max(scale_h, scale_w);
    scale_ = max_scale;
    int scaled_h = std::round(img.h / max_scale - 0.01f);
    int scaled_w = std::round(img.w / max_scale - 0.01f);
    CHECK_LE(scaled_h, netin_h_);
    CHECK_LE(scaled_w, netin_w_);
    if (netin_h_ > scaled_h) {
        pad_t = (netin_h_ - scaled_h) / 2;
        pad_l = 0;
    } else if (netin_w_ > scaled_w) {
        pad_t = 0;
        pad_l = (netin_w_ - scaled_w) / 2;;
    } else {
        pad_t = 0;
        pad_l = 0;
    }
    decoder_param.pad_l = pad_l;
    decoder_param.pad_t = pad_t;
    decoder_param.scale = scale_;
    // LOG(INFO) << "resize to w:" << scaled_w << " h: " << scaled_h;
    // LOG(INFO) << "scale:" << scale_;
    // LOG(INFO) << "pad l:" << pad_l;
    // LOG(INFO) << "pad t:" << pad_t;
    resized_img_.w = scaled_w;
    resized_img_.h = scaled_h;
    resized_img_.ts = img.ts;
    CHECK(img.format == resized_img_.format);
    CUDA_CHECK(cudaResize(
        img.ptr, img.w, img.h,
        resized_img_.ptr, resized_img_.w, resized_img_.h,
        img.format, cudaFilterMode::FILTER_POINT, stream
    ));
    // resized_image_buffer_->SaveTo("resized_img_buff.dat");

    netin_img_.ts = img.ts;
    CUDA_CHECK(cudaDrawRect(
            netin_img_.ptr, netin_img_.w, netin_img_.h, netin_img_.format,
            0, 0, netin_img_.w, netin_img_.h,
            {114, 114, 114, 255},
        {0, 0, 0, 0}, 0.f, stream
        ));
    // netin_buffer_->SaveTo("filled_img_buff.dat");
    CHECK(resized_img_.format == netin_img_.format);
    CUDA_CHECK(cudaOverlay(
        resized_img_.ptr, resized_img_.w, resized_img_.h,
        netin_img_.ptr, netin_img_.w, netin_img_.h,
        netin_img_.format, pad_l, pad_t, stream
    ));
    // netin_buffer_->SaveTo("overlay_img_buff.dat");
    CUDA_CHECK(cudaNormalize(
        netin_img_.ptr, {0, 255},
        netin_img_.ptr, {0, 1},
        netin_img_.w, netin_img_.h, netin_img_.format, stream
    ));
    // netin_buffer_->SaveTo("normed_img_buff.dat");
    infer_->infer(stream);
    decoder(decoder_param, stream);
    // netout_buffer_->SaveTo("netout.dat");
    CUDA_CHECK(cudaMemcpyAsync(decoder_box_nb_h_buffer_->GetPtr(), decoder_box_nb_d_buffer_->GetPtr(), sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    nvtxRangePop();
}

float IoU(const Box& b1, const Box& b2) {
    float x1 = std::max(b1.x0, b2.x0);
    float y1 = std::max(b1.y0, b2.y0);
    float x2 = std::min(b1.x1, b2.x1);
    float y2 = std::min(b1.y1, b2.y1);

    float intersectW = std::max(0.0f, x2 - x1);
    float intersectH = std::max(0.0f, y2 - y1);
    float intersectArea = intersectW * intersectH;

    float area1 = (b1.x1-b1.x0) * (b1.y1-b1.y0);
    float area2 = (b2.x1-b2.x0) * (b2.y1-b2.y0);
    float unionArea = area1 + area2 - intersectArea;

    if (unionArea <= 0.0f) {
        return 0.0f;
    }
    return intersectArea / unionArea;
}

void categoryNms(const Box* ptr, int box_nb, float iou_threshold, std::vector<Box>& ret) {
    std::unordered_map<int, std::vector<Box>> boxesByType;
    boxesByType.reserve(box_nb);
    for (size_t i = 0; i < box_nb; ++i) {
        boxesByType[ptr[i].type].push_back(ptr[i]);
    }
    ret.clear();
    ret.reserve(box_nb);
    for (auto& kv : boxesByType) {
        auto& boxes = kv.second;

        std::sort(boxes.begin(), boxes.end(), [](const Box& a, const Box& b){
            return a.score > b.score;
        });

        std::vector<Box> keep;
        keep.reserve(boxes.size());

        for (const auto& box : boxes) {
            bool shouldDiscard = false;
            for (const auto& keptBox : keep) {
                if (IoU(box, keptBox) > iou_threshold) {
                    shouldDiscard = true;
                    break;
                }
            }
            if (!shouldDiscard) {
                keep.push_back(box);
            }
        }
        ret.insert(ret.end(), keep.begin(), keep.end());
    }
}

void Stage::parseResults() {
    nvtxRangePushA("Stage::parseResults");
    int32_t nb = *reinterpret_cast<int32_t*>(decoder_box_nb_h_buffer_->GetPtr());
    LOG(INFO) << "raw det nb: " << nb;
    const Box* boxes = reinterpret_cast<const Box*>(decoder_boxes_buffer_->GetPtr());
    categoryNms(boxes, nb, nms_thr_, result_.boxes);
    nvtxRangePop();
}

Frame Stage::genRenderImage(cudaStream_t stream, bool mock) {
    if (mock) {
        return raw_img_;
    }
    nvtxRangePushA("Stage::genRenderImage");
    LOG(INFO) << "final det nb: " << result_.boxes.size();
    for (auto& box : result_.boxes) {
        CUDA_CHECK(cudaDrawRect(
            raw_img_.ptr, raw_img_.w, raw_img_.h, raw_img_.format,
            box.x0, box.y0, box.x1, box.y1,
            {0, 0, 0, 0},
            {0, 255, 0, 255}, 1.f, stream
        ));
    }
    nvtxRangePop();
    return raw_img_;
}

