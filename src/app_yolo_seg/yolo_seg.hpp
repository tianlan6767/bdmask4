#ifndef YOLO_SEG_HPP
#define YOLO_SEG_HPP

#include <vector>
#include <memory>
#include <string>
#include <future>
#include <opencv2/opencv.hpp>
#include <common/trt_tensor.hpp>
#include <common/result_detector.hpp>

namespace YoloSeg{

    using namespace std;
    using namespace ResultDetector;

    enum class NMSMethod : int{
        CPU = 0,         // General, for estimate mAP
        FastGPU = 1      // Fast NMS with a small loss of accuracy in corner cases
    };

    void image_to_tensor(const cv::Mat& image, shared_ptr<TRT::Tensor>& tensor, int ibatch);

    class Infer{
    public:
        virtual shared_future<BoxArray> commit(const cv::Mat& image) = 0;
        virtual vector<shared_future<BoxArray>> commits(const vector<cv::Mat>& images) = 0;
    };

    shared_ptr<Infer> create_infer(
        const string& engine_file, int gpuid,
        float confidence_threshold=0.25f, float nms_threshold=0.5f,
        NMSMethod nms_method = NMSMethod::FastGPU, int max_objects = 1024,
        bool use_multi_preprocess_stream = false
    );

}; // namespace YoloSeg

#endif // YOLO_SEG_HPP