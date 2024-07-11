#ifndef INFERAPP_HPP
#define INFERAPP_HPP

#include <memory>
#include <string>
#include <opencv2/opencv.hpp>
#include "app_fcos/fcos.hpp"
#include "app_yolo_seg/yolo_seg.hpp"
#include "app_anomalib/efficientad.hpp"

template<typename InferType, typename BoxArrayType>
class InferApp {
public:
    bool init(std::shared_ptr<void>& infer, const std::string& engine_path, float* mean, float* std, int device_id = 0, const char* method = "");
    BoxArrayType app(std::shared_ptr<void> infer, cv::Mat& image);
};

template<typename InferType, typename BoxArrayType>
bool InferApp<InferType, BoxArrayType>::init(std::shared_ptr<void>& infer, const std::string& engine_path, float* mean, float* std, int device_id, const char* method) {
    std::string methodStr(method); // 将C风格字符串转换为std::string
    if (methodStr == "blendmask") {
        std::shared_ptr<Fcos::Infer> fcosInfer = Fcos::create_infer(engine_path, device_id, 0.09, mean, std);
        infer = std::static_pointer_cast<void>(fcosInfer);
    }
    else if (methodStr == "yoloseg") {
        std::shared_ptr<YoloSeg::Infer> yoloSegInfer = YoloSeg::create_infer(engine_path, device_id, 0.25);
        infer = std::static_pointer_cast<void>(yoloSegInfer);
    }

    else if (methodStr == "efficientad") {
        std::shared_ptr<EfficientAd::Infer> efficientadInfer = EfficientAd::create_infer(engine_path, device_id, 100.0f);
        infer = std::static_pointer_cast<void>(efficientadInfer);
    }


    if (infer != nullptr) {
        return true;
    }
    else {
        return false;
    }
}


template <typename InferType, typename BoxArrayType>
BoxArrayType InferApp<InferType, BoxArrayType>::app(std::shared_ptr<void> infer, cv::Mat& image) {
    if (InferType* insInfer = static_cast<InferType*>(infer.get())) {
        auto boxes = insInfer->commit(image).get();
        return boxes;
    }
    else {
        std::cout << "Invalid 'infer' pointer" << std::endl;
        // 或者使用日志库记录错误消息
    }
}

#endif  // INFERAPP_HPP