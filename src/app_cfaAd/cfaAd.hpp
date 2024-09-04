#ifndef CFAAD_HPP
#define CFAAD_HPP

#include <vector>
#include <memory>
#include <string>
#include <future>
#include <opencv2/opencv.hpp>
#include <common/result_detector.hpp>


namespace CfaAd
{
    
    #define KERNEL_SIZE 33
    #define SIGMA 4.0f
    using namespace std;
    using namespace ResultDetector;

    enum class NMSMethod : int{
        CPU = 0,         // General, for estimate mAP
        FastGPU = 1      // Fast NMS with a small loss of accuracy in corner cases
    };

    class Infer
    {
    public:
        virtual shared_future<BoxArray> commit(const cv::Mat &image) = 0;
        virtual vector<shared_future<BoxArray>> commits(const vector<cv::Mat> &images) = 0;
    };

    shared_ptr<Infer> create_infer(const string &engine_file, SpliceInfoArray &spliceInfoArray, int gpuid = 0, float confidence_threshold = 0.5f,float mean[3]= {0,}, float std[3]={0,});

}; // namespace CfaAd

#endif // CFAAD_HPP