#include <assert.h>
#include <algorithm>
#include <cstdio>
#include <iomanip>
#include <boost/filesystem.hpp>

#include <common/ilogger.hpp>
#include <opencv2/opencv.hpp>

#include "bdmapp.h"
#include "app_fcos/fcos.hpp"
#include "nlohmann/json.hpp"

using namespace std;
using json = nlohmann::json;

static const char* cocolabels[] = {
     "1",  "2",  "3",  "4",  "5",  "6",  "7",  "8",  "9", "10",
    "11", "12", "13", "14", "15", "16", "17", "18", "19", "20",
    "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", 
    "31", "32", "33", "34", "35", "36", "37", "38", "39", "40", 
    "41", "42", "43", "44", "45", "46", "47", "48", "49", "50",
};

int main(){
    string fcos_engine_path = R"(/media/ps/data/train/LQ/task/bdm/bdmask/workspace/code/trt/model/model_0413999-orig-fp16)";
    BdmApp bdmapp;
    shared_ptr<Fcos::Infer> fcos1 = nullptr;
    int device_id1 = 0;
    float mean[] = {90};
    float std[] = {77};

    bool result1 = bdmapp.bdminit(fcos1, fcos_engine_path, mean, std, device_id1);

    string src = R"(/media/ps/data/train/LQ/task/bdm/bdmask/workspace/code/trt/data/data2/val/*.jpg)";
    string dst = R"(/media/ps/data/train/LQ/task/bdm/bdmask/workspace/code/trt/data/data2/inf/new1/ptq-all-train1111110)";

    vector<cv::String> files_;
    files_.reserve(10000);

    cv::glob(src, files_, true);
    vector<string> files(files_.begin(), files_.end());
    vector<float> avg_times1;
    // warm-up
    for(int i = 0; i < 1; ++i){
        cv::Mat image(2048, 2048, CV_8UC1);
        image.setTo(128);
        fcos1->commit(image);
    }
    assert(result1);
    int num_imp = 1;
    int noc = 1;
    int noc_num = noc;
    int batch_size = 5;
    json resjson;
    // 多图推理
    // while(noc){
    //     for(int im_idx= 0; im_idx < files.size(); im_idx += batch_size){
    //         vector<cv::Mat> images;
    //         vector<string> imns;
    //         for(int n = 0; n < batch_size; n++){
    //             if (im_idx + n >= files.size()) break;
    //             cv::Mat image = cv::imread(files[im_idx + n], -1);
    //             boost::filesystem::path path(files[im_idx + n]);
    //             string imn = path.stem().string();
    //             string nimp_result = dst + "/" + imn  + ".jpg";
    //             images.push_back(image);
    //             imns.push_back(imn);
    //         }
    //         auto begin_time1 = iLogger::timestamp_now_float();
    //         // printf("当前提交图像******\n");
    //         auto images_boxes = fcos1->commits(images);
    //         auto end_time1 = iLogger::timestamp_now_float();
    //         avg_times1.emplace_back(end_time1 - begin_time1);
    //         for(int image_id= 0; image_id < images_boxes.size(); ++image_id){
    //             json regionsarray = json::array();
    //             auto imn = imns[image_id];
    //             auto image = images[image_id].clone();

    //             auto image_start_time1 = iLogger::timestamp_now_float();
    //             // printf("当前开始get数据********\n");
    //             auto boxes = images_boxes[image_id].get();
    //             auto image_end_time1 = iLogger::timestamp_now_float();
    //             avg_times1.emplace_back(image_end_time1 - image_start_time1);
    //             int box_idx = 0;
    //             for(auto & box : boxes){
    //                 cv::Scalar color(0, 255, 0);
    //                 cv::rectangle(image, cv::Point(box.left, box.top), cv::Point(box.right, box.bottom), color, 3);
    //                 auto name      = cocolabels[box.class_label];
    //                 auto caption   = cv::format("%s %.2f", name, sqrt(box.confidence));
    //                 int text_width = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
    
    //                 printf("%d  %f  %f  %f  %f  %f\n", box.class_label, box.left, box.top, box.right, box.bottom, sqrt(box.confidence));
    //                 cv::rectangle(image, cv::Point(box.left-3, box.top-33), cv::Point(box.left + text_width, box.top), color, -1);
    //                 cv::putText(image, caption, cv::Point(box.left, box.top-5), 0, 1, cv::Scalar::all(0), 2, 16);

    //                 if (box.seg) {
    //                     string nimp_result_mask = dst + "/" + imn +"_mask_" + to_string(box_idx) +".jpg";
    //                     auto box_mask = cv::Mat(box.seg->height, box.seg->width, CV_8U, box.seg->data);
    //                     // cv::imwrite(nimp_result_mask,
    //                     //             box_mask);

    //                     cv::Mat edges;
    //                     std::vector<std::vector<cv::Point>> contours;
    //                     std::vector<cv::Vec4i> hierarhy;
    //                     cv::findContours(box_mask, contours, hierarhy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    //                     cv::Point topleft(box.left, box.top);


    //                     // mask里面的轮廓
    //                     for (auto& contour:contours){
    //                         for (auto& point:contour){
    //                             point += topleft;
    //                         }
    //                     }
    //                     // 创建一个空的数组
    //                     json allPointsX = json::array();
    //                     json allPointsY = json::array();
    //                     json j1;
    //                     for (auto& contour:contours){
    //                         // json j1;
    //                         if(contour.size() < 3) continue;
    //                         for(auto& point:contour){
    //                             allPointsX.push_back(point.x);
    //                             allPointsY.push_back(point.y);
    //                         }
    //                     }
    //                     j1["region_attributes"] = {
    //                         {"regions", to_string(box.class_label + 1)},
    //                         {"score",  std::round(sqrt(box.confidence) * 10000.0) / 10000.0}
    //                     };
    //                     j1["shape_attributes"] = {
    //                         {"all_points_x", allPointsX},
    //                         {"all_points_y", allPointsY}
    //                     };
    //                     regionsarray.push_back(j1);
    //                     cv::drawContours(image, contours, -1, cv::Scalar(0, 0, 255), 2);
    //                     }
    //                     ++box_idx;
    //                 }
    //             resjson[imn] = {
    //                     {"filename", imn},
    //                     {"regions", regionsarray},
    //                     {"type", "inf"}
    //                 };
                
    //             string nimp_result = dst + "/" + imn  + ".jpg";
    //             printf("***********第%d張圖片---当前%s, 图片有%d个缺陷, 当前推理 %d 张图片耗时平均:%f-%f-%f***********\n", num_imp,
    //                         imn.c_str(), boxes.size(), (images_boxes.size()), (end_time1 - begin_time1), (end_time1 - begin_time1)/(images_boxes.size()), (image_end_time1 - image_start_time1));
    //             // cv::imwrite(nimp_result, image);
    //             ++num_imp;

    //             }
    //         }
    //     noc--;
    // }
    

    // 单图推理
    while(noc){
        for(int im_idx=0; im_idx < files.size(); ++im_idx){
            cv::Mat image = cv::imread(files[im_idx], -1);
            boost::filesystem::path path(files[im_idx]);
            string imn = path.stem().string() + ".jpg";
            string nimp_result = dst + "/" + path.stem().string()  + ".jpg";
            auto begin_time1 = iLogger::timestamp_now_float();
            auto defect_res = bdmapp.bdmapp(fcos1,image);
            auto end_time1 = iLogger::timestamp_now_float();

            cv::cvtColor(image, image, cv::COLOR_GRAY2BGR);
            // 绘制到图片上
            int idx = 0;
            json regionsarray = json::array();
            for(auto & box : defect_res){
                
                cv::Scalar color(0, 255, 0);
                cv::rectangle(image, cv::Point(box.left, box.top), cv::Point(box.right, box.bottom), color, 3);
                auto name      = cocolabels[box.class_label];
                auto caption   = cv::format("%s %.2f", name, sqrt(box.confidence));
                int text_width = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
 
                printf("%d  %f  %f  %f  %f  %f\n", box.class_label, box.left, box.top, box.right, box.bottom, sqrt(box.confidence));
                cv::rectangle(image, cv::Point(box.left-3, box.top-33), cv::Point(box.left + text_width, box.top), color, -1);
                cv::putText(image, caption, cv::Point(box.left, box.top-5), 0, 1, cv::Scalar::all(0), 2, 16);

                if (box.seg) {
                    string nimp_result_mask = dst + "/" + path.stem().string() +"_mask_" + to_string(idx) +".jpg";
                    auto box_mask = cv::Mat(box.seg->height, box.seg->width, CV_8U, box.seg->data);
                    // cv::imwrite(nimp_result_mask,
                    //             box_mask);

                    cv::Mat edges;
                    std::vector<std::vector<cv::Point>> contours;
                    std::vector<cv::Vec4i> hierarhy;
                    cv::findContours(box_mask, contours, hierarhy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
                    cv::Point topleft(box.left, box.top);


                    // mask里面的轮廓
                    for (auto& contour:contours){
                        for (auto& point:contour){
                            point += topleft;
                        }
                    }
                    // 创建一个空的数组
                    // json j1;
                    // json allPointsX = json::array();
                    // json allPointsY = json::array();
                    
                    for (auto& contour:contours){

                        // 创建一个空的数组
                        json j1;
                        json allPointsX = json::array();
                        json allPointsY = json::array();

                        if(contour.size() < 3) continue;
                        for(auto& point:contour){
                            allPointsX.push_back(point.x);
                            allPointsY.push_back(point.y);
                        }
                        j1["region_attributes"] = {
                            {"regions", to_string(box.class_label + 1)},
                            {"score",  std::round(sqrt(box.confidence) * 10000.0) / 10000.0}
                        };
                        j1["shape_attributes"] = {
                            {"all_points_x", allPointsX},
                            {"all_points_y", allPointsY}
                        };
                        regionsarray.push_back(j1);
                    }

                    // j1["region_attributes"] = {
                    //     {"regions", to_string(box.class_label + 1)},
                    //     {"score",  std::round(sqrt(box.confidence) * 10000.0) / 10000.0}
                    // };
                    // j1["shape_attributes"] = {
                    //     {"all_points_x", allPointsX},
                    //     {"all_points_y", allPointsY}
                    // };
                    // regionsarray.push_back(j1);

                    cv::drawContours(image, contours, -1, cv::Scalar(0, 0, 255), 2);
                    idx++;
                    }
            }
            resjson[imn] = {
                {"filename", imn},
                {"regions", regionsarray},
                {"type", "inf"}
            };


            printf("***********第%d張圖片---当前%s, 图片有%d个缺陷, 当前推理耗时:%f***********\n", num_imp,
                        imn.c_str(), defect_res.size(), end_time1 - begin_time1);

            cv::imwrite(nimp_result, image);
            ++num_imp;

            avg_times1.emplace_back(end_time1 - begin_time1);
        }
        noc--;
    }
    string json_file = dst + "/data-trt-ptq-all-train8000.json"; 
    std::ofstream file(json_file);
    if (file.is_open()) {
        file << resjson.dump(4);
        file.close();
        std::cout << "JSON data saved to file." << std::endl;
    } else {
        std::cerr << "Unable to open file." << std::endl;
    }
    
    float sum = 0;
    for (int i = 0; i < avg_times1.size(); i++) {
        sum += avg_times1[i];
    };

    printf("第一张卡共测试%d张, 总耗时:%.4f秒, 图片平均耗时: %.4f毫秒\n", files.size() * noc_num, sum/1000, (sum / (files.size()*noc_num)));
    printf("success\n");
    return 0;
}