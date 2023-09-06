#ifndef OBJECT_DETECTOR_HPP
#define OBJECT_DETECTOR_HPP

#include <vector>
#include <opencv2/opencv.hpp>

namespace ObjectDetector
{
    using namespace std;

    struct Obj
    {
        float left, top, right, bottom, confidence;
        int class_label;
        float top_feat[784];
        Obj() = default;

        Obj(float left, float top, float right, float bottom, float confidence, int class_label, float *top_feat)
            : left(left), top(top), right(right), bottom(bottom), confidence(confidence), class_label(class_label) {
                for (int i = 0; i < 784; i++) {
                    this->top_feat[i] = top_feat[i];
                }
            }
    };
    
    struct Defect
    {
        vector<int> labels{};
        vector<float> scores{};
        vector<cv::Mat> masks{};
        vector<Obj> boxes{};
    };



    struct Base {
        static const int kWidth = 768;
        static const int kHeight = 1024;
        std::shared_ptr<float> base;
        Base() : base(new float[kWidth * kHeight], std::default_delete<float[]>()) {}
        Base(float val) : base(new float[kWidth * kHeight], std::default_delete<float[]>())
        {
            std::fill(base.get(), base.get() + kWidth * kHeight, val);
        }
    };


    struct Result
    {
        std::vector<Obj> BoxArray;
        std::vector<Base> BasesArray;
    };
    
    typedef Result ResultArray;

};

#endif // OBJECT_DETECTOR_HPP