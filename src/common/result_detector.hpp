#ifndef RESULT_DETECTOR_HPP
#define RESULT_DETECTOR_HPP

#include <vector>
#include <memory>
#include <common/cuda_tools.hpp>


namespace ResultDetector{
    struct InstanceSegmentMap {
            int width = 0, height = 0;      // width % 8 == 0
            int left = 0, top = 0;          // 160x160 feature map
            unsigned char *data = nullptr;  // is width * height memory
            InstanceSegmentMap(int width, int height){
                this->width  = width;
                this->height = height;
                checkCudaRuntime(cudaMallocHost(&this->data, width * height));
            };
            virtual ~InstanceSegmentMap(){
                if(this->data){
                    checkCudaRuntime(cudaFreeHost(this->data));
                    this->data = nullptr;
                }
                this->width  = 0;
                this->height = 0;
            };
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
};

#endif // RESULT_DETECTOR_HPP