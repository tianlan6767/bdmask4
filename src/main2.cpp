#include <stdio.h>
#include <string.h>
#include <common/ilogger.hpp>
#include <functional>
#include <app.hpp>

static const char* cocolabels[] = {
    "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
    "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
    "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass",
    "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
};


static void draw_mask(cv::Mat& image, YoloSeg::Box& obj, cv::Scalar& color){

    // compute IM
    float scale_x = 640 / (float)image.cols;
    float scale_y = 640 / (float)image.rows;
    float scale   = std::min(scale_x, scale_y);
    float ox      = -scale * image.cols * 0.5 + 640 * 0.5 + scale * 0.5 - 0.5;
    float oy      = -scale * image.rows * 0.5 + 640 * 0.5 + scale * 0.5 - 0.5;
    cv::Mat M     = (cv::Mat_<float>(2, 3)<<scale, 0, ox, 0, scale, oy);    
    
    cv::Mat IM;
    cv::invertAffineTransform(M, IM); 

    cv::Mat mask_map = cv::Mat::zeros(cv::Size(512, 512), CV_8UC1);
    cv::Mat small_mask(obj.seg->height, obj.seg->width, CV_8UC1, obj.seg->data);
    cv::Rect roi(obj.seg->left, obj.seg->top, obj.seg->width, obj.seg->height);
    small_mask.copyTo(mask_map(roi));
    cv::resize(mask_map, mask_map, cv::Size(640, 640)); // 640x640
    cv::threshold(mask_map, mask_map, 128, 1, cv::THRESH_BINARY);

    cv::Mat mask_resized;
    cv::warpAffine(mask_map, mask_resized, IM, image.size(), cv::INTER_LINEAR);

    // create color mask
    cv::Mat colored_mask = cv::Mat::ones(image.size(), CV_8UC3);
    colored_mask.setTo(color);

    cv::Mat masked_colored_mask;
    cv::bitwise_and(colored_mask, colored_mask, masked_colored_mask, mask_resized);

    // create mask indices
    cv::Mat mask_indices;
    cv::compare(mask_resized, 1, mask_indices, cv::CMP_EQ);
    
    cv::Mat image_masked, colored_mask_masked;
    image.copyTo(image_masked, mask_indices);
    masked_colored_mask.copyTo(colored_mask_masked, mask_indices);

    // weighted sum
    cv::Mat result_masked;
    cv::addWeighted(image_masked, 0.6, colored_mask_masked, 0.4, 0, result_masked);
    
    // copy result to image
    result_masked.copyTo(image, mask_indices);
}



int main(){
    InferApp<YoloSeg::Infer, ResultDetector::BoxArray> yolosegapp;
    InferApp<Fcos::Infer, ResultDetector::BoxArray> fcosapp;
    std::shared_ptr<void> yoloseginfer=nullptr;
    std::shared_ptr<void> fcosinfer = nullptr;
    int device_id = 1;
    float mean[] = {85.18,85.18,85.18};
    float std[] = {72.56,72.56,72.56};
    std::string model_file = R"(/media/ps/data/train/LQ/task/yolo/ultralytics-8.1.0/runs/segment/train7-m-seg/weights/epoch300.transd.engine)";
    const char* method = "blendmask";
    std::string fcos_engine_path = R"(/media/ps/data/train/LQ/task/bdm/bdmask/workspace/tmp/imgs2/model_1573999.trtmodel)";
    bool result_yoloseg = yolosegapp.init(yoloseginfer, model_file, mean, std, device_id, "yoloseg");
    bool result_fcos = fcosapp.init(fcosinfer, fcos_engine_path, mean, std, device_id, "blendmask");
    if(!result_yoloseg){
        std::cout << "初始化失败" <<std::endl;
        return 0;
    }

    std::string filename = R"(/media/ps/data/train/LQ/task/bdm/bdmask/workspace/tmp/imgs2/20240322201248.bmp)";
    cv::Mat image = cv::imread(filename);
    if(image.empty()){
        INFOE("Image is empty");
        return 0;
    }    

    // auto boxes = yolosegapp.app(yoloseginfer, image);
    auto boxes_fcos = fcosapp.app(fcosinfer, image);
    
    int i = 0;
    for(auto& obj : boxes_fcos){
        uint8_t b, g, r;
        std::tie(b, g, r) = iLogger::random_color(obj.class_label);
        cv::Scalar color(b, g, r);
        if(obj.seg){
            // draw_mask(image, obj, color);
            cv::imwrite(iLogger::format("%d_mask.jpg", i),
                        cv::Mat(obj.seg->height, obj.seg->width, CV_8U, obj.seg->data));
            i++;
        }
    }

    for(auto& obj : boxes_fcos){
        uint8_t b, g, r;
        std::tie(b, g, r) = iLogger::random_color(obj.class_label);
        cv::rectangle(image, cv::Point(obj.left, obj.top), cv::Point(obj.right, obj.bottom), cv::Scalar(b, g, r), 5);

        auto name    = cocolabels[obj.class_label];
        auto caption = iLogger::format("%s %.2f", name, obj.confidence);
        int width    = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
        cv::rectangle(image, cv::Point(obj.left-3, obj.top-33), cv::Point(obj.left + width, obj.top), cv::Scalar(b, g, r), -1);
        cv::putText(image, caption, cv::Point(obj.left, obj.top-5), 0, 1, cv::Scalar::all(0), 2, 16);
    }
    INFO("Save to Result-seg.jpg, %d objects", boxes_fcos.size());
    cv::imwrite("Result-seg.jpg", image);

    return 0;
}
