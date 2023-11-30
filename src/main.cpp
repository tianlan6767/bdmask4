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
    string fcos_engine_path = R"(/media/ps/data/train/LQ/task/bdm/bdmask/workspace/models/JR/JR_1124-dy-b1-5472)";
    BdmApp bdmapp;
    shared_ptr<Fcos::Infer> fcos1 = nullptr;
    int device_id1 = 0;
    float mean[] = {57.14, 55.92, 56.19};
    float std[] = {61.46, 61.27, 61.23};

    // float mean[] = {90};
    // float std[] = {77};

    bool result1 = bdmapp.bdminit(fcos1, fcos_engine_path, mean, std, device_id1);

    string src = R"(/media/ps/data/train/LQ/task/bdm/bdmask/workspace/models/JR/imgs_jpg_segment_jpg/*.jpg)";
    string dst = R"(/media/ps/data/train/LQ/task/bdm/bdmask/workspace/models/OQC/tinf-b5)";

    vector<cv::String> files_;
    files_.reserve(10000);

    cv::glob(src, files_, true);
    vector<string> files(files_.begin(), files_.end());
    vector<float> avg_times1_mul;

    // warm-up
    // for(int i = 0; i < 5; ++i){
    //     cv::Mat image(4096, 4096, CV_8UC3);
    //     image.setTo(128);
    //     fcos1->commit(image);
    // }
    
    assert(result1);
    int num_imp = 1;
    int noc = 1;
    int noc_num = noc;
    int batch_size = 3;
    json resjson;
    // // 多图推理
    // while(noc){
    //     for(int im_idx= 0; im_idx < files.size(); im_idx += batch_size){
    //         vector<cv::Mat> images;
    //         vector<string> imns;
    //         for(int n = 0; n < batch_size; n++){
    //             if (im_idx + n >= files.size()) break;
    //             cv::Mat image = cv::imread(files[im_idx + n], 1);
    //             boost::filesystem::path path(files[im_idx + n]);
    //             string imn = path.stem().string() + ".jpg";
    //             string nimp_result = dst + "/" + imn  + ".jpg";
    //             images.push_back(image);
    //             imns.push_back(imn);
    //         }
    //         auto begin_time1_mul = iLogger::timestamp_now_float();
    //         auto images_boxes = fcos1->commits(images);
    //         vector<Fcos::BoxArray> imgs_boxes = {};
    //         for(int image_id= 0; image_id < images_boxes.size(); ++image_id){
    //             auto img_boxes = images_boxes[image_id].get();
    //             imgs_boxes.emplace_back(img_boxes);
    //         }
    //         auto end_time1_mul = iLogger::timestamp_now_float();
    //         avg_times1_mul.emplace_back(end_time1_mul - begin_time1_mul);
    //         for(int image_id= 0; image_id < imgs_boxes.size(); ++image_id){
    //             json regionsarray = json::array();
    //             auto imn = imns[image_id];
    //             auto image = images[image_id].clone();
    //             auto boxes= imgs_boxes[image_id];
    //             int box_idx = 0;
    //             // cv::cvtColor(image, image, cv::COLOR_GRAY2BGR);
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

    //                     for (auto& contour:contours){

    //                         // 创建一个空的数组
    //                         json j1;
    //                         json allPointsX = json::array();
    //                         json allPointsY = json::array();
    //                         if(contour.size() < 3) continue;
    //                         for(auto& point:contour){
    //                             allPointsX.push_back(point.x);
    //                             allPointsY.push_back(point.y);
    //                         }
    //                         j1["region_attributes"] = {
    //                         {"regions", to_string(box.class_label + 1)},
    //                         {"score",  std::round(sqrt(box.confidence) * 10000.0) / 10000.0}
    //                         };
    //                         j1["shape_attributes"] = {
    //                             {"all_points_x", allPointsX},
    //                             {"all_points_y", allPointsY}
    //                         };
    //                         regionsarray.push_back(j1);
    //                     }
                        
    //                     cv::drawContours(image, contours, -1, cv::Scalar(0, 0, 255), 2);
    //                     ++box_idx;
    //                     }
    //             }
    //             resjson[imn] = {
    //                     {"filename", imn},
    //                     {"regions", regionsarray},
    //                     {"type", "inf"}
    //             };
                
    //             string nimp_result = dst + "/" + imn  + ".jpg";
    //             printf("***********第%d張圖片---当前%s, 图片有%d个缺陷, 当前推理 %d 张图片耗时平均:%.4f-%.4f***********\n", num_imp,
    //                         imn.c_str(), boxes.size(), images_boxes.size(), (end_time1_mul - begin_time1_mul), ((end_time1_mul - begin_time1_mul)/images_boxes.size()));
    //             cv::imwrite(nimp_result, image);
    //             ++num_imp;
    //             }
    //         }
    //     noc--;
    // }
        
    // 单图推理
    int noc_single = 1;
    vector<float> avg_times1_single;
    while(noc_single){
        for(int im_idx=0; im_idx < files.size(); ++im_idx){
            cv::Mat image = cv::imread(files[im_idx], 0);
            boost::filesystem::path path(files[im_idx]);
            string imn = path.stem().string() + ".jpg";
            string nimp_result = dst + "/" + path.stem().string()  + ".jpg";
            auto begin_time1_single = iLogger::timestamp_now_float();
            auto defect_res = bdmapp.bdmapp(fcos1,image);
            auto end_time1_single = iLogger::timestamp_now_float();
            avg_times1_single.emplace_back(end_time1_single - begin_time1_single);

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
                        imn.c_str(), defect_res.size(), end_time1_single - begin_time1_single);

            cv::imwrite(nimp_result, image);
            ++num_imp;

        }
        noc_single--;
    }

    string json_file = dst + "/data.json"; 
    std::ofstream file(json_file);
    if (file.is_open()) {
        file << resjson.dump(4);
        file.close();
        std::cout << "JSON data saved to file." << std::endl;
    } else {
        std::cerr << "Unable to open file." << std::endl;
    }
    
    // float sum_mul = 0;
    // for(int i = 1; i < avg_times1_mul.size(); i++) {
    //     sum_mul += avg_times1_mul[i];
    // };

    float sum_single = 0;
    for (int i = 1; i < avg_times1_single.size(); i++) {
        sum_single += avg_times1_single[i];
    };

    // printf("%d 共测试 %d 张,  多batch总耗时:%.4f秒, 图片平均耗时: %.4f毫秒\n", avg_times1_mul.size(), files.size() * noc_num, sum_mul/1000, (sum_mul / (files.size()*noc_num - batch_size)));
    printf("共测试%d张, 单batch总耗时:%.4f秒, 图片平均耗时: %.4f毫秒\n", files.size() * noc_num, sum_single/1000, (sum_single / (files.size()*noc_num -1)));
    printf("success\n");
    return 0;
}