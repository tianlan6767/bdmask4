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
    float left, top, right, bottom, confidence, meanval, maxval;
    int class_label;
    std::shared_ptr<InstanceSegmentMap> seg;  // valid only in segment task
    
    Box() = default;
    Box(float left, float top, float right, float bottom, float confidence, int class_label, float meanval, float maxval)
        : left(left),
            top(top),
            right(right),
            bottom(bottom),
            confidence(confidence),
            class_label(class_label),
            meanval(meanval), 
            maxval(maxval) {}
    };


    struct SpliceInfo{
        std::string image_name;
        int crop_x, crop_y, splice_x, splice_y, crop_w, crop_h, region_idx;
        SpliceInfo() = default;
        SpliceInfo(std::string image_name, int crop_x, int crop_y, int splice_x, int splice_y, int crop_w, int crop_h, int region_idx)
        : image_name(image_name),
            crop_x(crop_x),
            crop_y(crop_y),
            splice_x(splice_x),
            splice_y(splice_y),
            crop_w(crop_w),
            crop_h(crop_h),
            region_idx(region_idx) {}
    };
    
    typedef std::vector<SpliceInfo> SpliceInfoArray;
    typedef std::vector<Box> BoxArray;
};

#endif // RESULT_DETECTOR_HPP