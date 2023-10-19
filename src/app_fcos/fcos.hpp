#ifndef FCOS_HPP
#define FCOS_HPP

#include <vector>
#include <memory>
#include <string>
#include <future>
#include <opencv2/opencv.hpp>
#include "../common/object_detector.hpp"

namespace Fcos
{

    using namespace std;
    using namespace ObjectDetector;

    enum class NMSMethod : int{
        CPU = 0,         // General, for estimate mAP
        FastGPU = 1      // Fast NMS with a small loss of accuracy in corner cases
    };

    
    struct InstanceSegmentMap {
        int width = 0, height = 0;      // width % 8 == 0
        unsigned char *data = nullptr;  // is width * height memory

        InstanceSegmentMap(int width, int height);
        virtual ~InstanceSegmentMap();
    };

    struct Box {
    float left, top, right, bottom, confidence;
    int class_label;
    std::shared_ptr<InstanceSegmentMap> seg;  // valid only in segment task

    Box() = default;
    Box(float left, float top, float right, float bottom, float confidence, int class_label)
        : left(left),
            top(top),
            right(right),
            bottom(bottom),
            confidence(confidence),
            class_label(class_label) {}
    };

    typedef std::vector<Box> BoxArray;


    class Infer
    {
    public:
        virtual shared_future<BoxArray> commit(const cv::Mat &image) = 0;
        virtual vector<shared_future<BoxArray>> commits(const vector<cv::Mat> &images) = 0;
    };

    shared_ptr<Infer> create_infer(const string &engine_file, int gpuid = 0, float confidence_threshold = 0.09f, float mean[]={0,}, float std[]={0,}, float nms_threshold = 0.6f, NMSMethod nms_method = NMSMethod::FastGPU);

}; // namespace Fcos

#endif // FCOS_HPP