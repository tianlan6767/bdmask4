#include <stdio.h>
#include <string.h>
#include <common/ilogger.hpp>
#include <functional>
#include <app.hpp>
#include <boost/filesystem.hpp>

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
    InferApp<YoloSeg::Infer, ResultDetector::BoxArray> yoloseg_app;
    InferApp<Fcos::Infer, ResultDetector::BoxArray> fcos_app;
    InferApp<EfficientAd::Infer, ResultDetector::BoxArray> effad_app;
    std::shared_ptr<void> yoloseg_infer=nullptr;
    std::shared_ptr<void> fcos_infer = nullptr;
    std::shared_ptr<void> effad_infer = nullptr;
    int device_id = 0;

    // *************测试初始化耗时*************
    // int num = 1000;
    // while (num)
    // {   
    //     auto begin_time1_1 = iLogger::timestamp_now_float();
    //     bool result_fcos = fcosapp.init(fcosinfer, fcos_engine_path, mean, std, device_id, "blendmask");
    //     auto begin_time1_2 = iLogger::timestamp_now_float();
    //     printf("当前模型加载耗时:%f***********\n", begin_time1_2 - begin_time1_1);
    //     auto begin_time1_3 = iLogger::timestamp_now_float();
    //     fcosinfer.reset();
    //     auto begin_time1_4 = iLogger::timestamp_now_float();
    //     printf("当前推理释放耗时:%f***********\n", begin_time1_4 - begin_time1_3);
    // }

    float mean[] = {0,0,0};
    float std[] = {0,0,0};
    
    // ********模型初始化**************
    // std::string model_file = R"(/media/ps/data/train/LQ/task/yolo/ultralytics/ultralytics/weights/yolov8m-seg.transd-dy-1024-2048-4096.trtmodel)";
    // std::string fcos_engine_path = R"(/media/ps/data1/train/LQ/task/bdm/bdmask/workspace/tmp/jt/model_0887999.trtmodel)";
    // bool result_yoloseg = yolosegapp.init(yoloseginfer, model_file, mean, std, device_id, "yoloseg");

    // bool result = fcosapp.init(fcosinfer, fcos_engine_path, mean, std, device_id, "blendmask");

    std::string effad_engine_path = R"(/media/ps/data1/train/LQ/task/bdm/bdmask/workspace/models/ead/new/export/model_simple.trtmodel)";
    bool result = effad_app.init(effad_infer, effad_engine_path, mean, std, device_id, "efficientad");

    if(!result){
        std::cout << "初始化失败" <<std::endl;
        return 0;
    }
    
    std::string src = R"(/media/ps/data1/train/LQ/task/bdm/bdmask/workspace/models/ead/new/images/*.png)";
    // std::string src = R"(/media/ps/data1/train/LQ/task/bdm/bdmask/workspace/models/ead/images/*.png)";
    std::string dst = R"(/media/ps/data1/train/LQ/task/bdm/bdmask/workspace/models/ead/new/inf4)";
    std::vector<cv::String> files_;
    files_.reserve(100000);

    cv::glob(src, files_, true);
    std::vector<std::string> files(files_.begin(), files_.end());
    std::vector<float> avg_times;
    int i = 0;

    for(int im_idx=0; im_idx < files.size(); ++im_idx){
        cv::Mat image = cv::imread(files[im_idx], 1);
        boost::filesystem::path path(files[im_idx]);
        std::string imn = path.stem().string() + "_mask.jpg";
        std::string nimp_result = dst + "/" + imn;
        if(image.empty()){
            INFOE("Image is empty");
            return 0;
        } 
        // cv::resize(image, image, dsize, 0, 0);
        
         // 获取图像的尺寸
        int width = image.cols;
        int height = image.rows;
        int channels = image.channels();

        // 使用 printf 打印图像的尺寸
        printf("Image size: %d x %d x %d\n", width, height, channels);
        auto begin_time1_mul = iLogger::timestamp_now_float();
        auto boxes = effad_app.app(effad_infer, image);
        auto end_time1_mul = iLogger::timestamp_now_float();
        for(auto& box : boxes){
            uint8_t b, g, r;
            std::tie(b, g, r) = iLogger::random_color(0);
            cv::Scalar color(b, g, r);
            if(box.seg){
                // draw_mask(image, box, color);
                // cv::imwrite(nimp_result,cv::Mat(box.seg->height, box.seg->width, CV_8U, box.seg->data));
                // i++;
                auto box_mask = cv::Mat(box.seg->height, box.seg->width, CV_8U, box.seg->data);
                cv::Mat edges;
                std::vector<std::vector<cv::Point>> contours;
                std::vector<cv::Vec4i> hierarhy;
                cv::findContours(box_mask, contours, hierarhy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
                cv::Point topleft(box.left, box.top);

                cv::drawContours(image, contours, -1, cv::Scalar(0, 0, 255), 2);
            }
        }
        cv::imwrite(nimp_result, image);
        // auto boxes = yolosegapp.app(yoloseginfer, image);
        // auto boxes_fcos = fcosapp.app(fcosinfer, image);


        avg_times.emplace_back(end_time1_mul - begin_time1_mul);
        printf("当前推理耗时:%f***********\n", end_time1_mul - begin_time1_mul);

    //     int i = 0;
        // for(auto& obj : boxes){
        //     uint8_t b, g, r;
        //     std::tie(b, g, r) = iLogger::random_color(obj.class_label);
        //     cv::Scalar color(b, g, r);
        //     if(obj.seg){
        //         // draw_mask(image, obj, color);
        //         cv::imwrite(iLogger::format("%d_mask.jpg", i),
        //                     cv::Mat(obj.seg->height, obj.seg->width, CV_8U, obj.seg->data));
        //         i++;
        //     }
        // }

    //     for(auto& obj : boxes){
    //         uint8_t b, g, r;
    //         std::tie(b, g, r) = iLogger::random_color(obj.class_label);
    //         cv::rectangle(image, cv::Point(obj.left, obj.top), cv::Point(obj.right, obj.bottom), cv::Scalar(b, g, r), 5);

    //         auto name    = cocolabels[obj.class_label];
    //         auto caption = iLogger::format("%s %.2f", name, obj.confidence);
    //         int width    = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
    //         cv::rectangle(image, cv::Point(obj.left-3, obj.top-33), cv::Point(obj.left + width, obj.top), cv::Scalar(b, g, r), -1);
    //         cv::putText(image, caption, cv::Point(obj.left, obj.top-5), 0, 1, cv::Scalar::all(0), 2, 16);
    //     }
    //     INFO("Save to Result-seg.jpg, %d objects", boxes.size());
    //     cv::imwrite("Result-seg.jpg", image);
    }
    float sum_mul = 0;
    for(int i = 1; i < avg_times.size(); i++) {
        sum_mul += avg_times[i];
    };
    printf("总图片数: %d, batch总耗时:%.4f秒, 图片平均耗时: %.4f毫秒\n",(files.size() - 1), sum_mul/1000, (sum_mul / (files.size() - 1)));

    return 0;
}
