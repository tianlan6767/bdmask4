#ifndef BDM_HPP
#define BDM_HPP

#include <vector>
#include <memory>
#include <string>
#include <future>
#include <opencv2/opencv.hpp>
#include "../common/object_detector.hpp"

namespace Bdm
{

    using namespace std;
    using namespace ObjectDetector;

    enum class NMSMethod : int{
        CPU = 0,         // General, for estimate mAP
        FastGPU = 1      // Fast NMS with a small loss of accuracy in corner cases
    };

    class Infer
    {
    public:
        virtual shared_future<MaskArray> commit(const cv::Mat &image) = 0;
        virtual vector<shared_future<MaskArray>> commits(const vector<cv::Mat> &images) = 0;
    };

    shared_ptr<Infer> create_infer(const string &fcos_file, const string &mask_file, int gpuid = 0, float confidence_threshold = 0.09f, float mean[]={0,}, float std[]={0,}, float nms_threshold = 0.6f, NMSMethod nms_method = NMSMethod::FastGPU);

}; // namespace Bdm

#endif // BDM_HPP