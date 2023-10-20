#include <assert.h>
#include <algorithm>
#include <cstdio>
#include <boost/filesystem.hpp>

#include <common/ilogger.hpp>
#include <opencv2/opencv.hpp>

#include "bdmapp.h"
#include "app_fcos/fcos.hpp"


using namespace std;

static const char* cocolabels[] = {
     "1",  "2",  "3",  "4",  "5",  "6",  "7",  "8",  "9", "10",
    "11", "12", "13", "14", "15", "16", "17", "18", "19", "20",
    "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", 
    "31", "32", "33", "34", "35", "36", "37", "38", "39", "40", 
    "41", "42", "43", "44", "45", "46", "47", "48", "49", "50",
};

int main(){
    string fcos_engine_path = R"(/media/ps/data/train/LQ/LQ/bdms/bdmask/workspace/models/model_0364999-dd)";
    BdmApp bdmapp;
    shared_ptr<Fcos::Infer> fcos1 = nullptr;
    int device_id1 = 1;
    float mean[] = {90};
    float std[] = {77};

    bool result1 = bdmapp.bdminit(fcos1, fcos_engine_path, mean, std, device_id1);

    string src = R"(/media/ps/data/train/LQ/LQ/bdms/bdmask/workspace/inffff/oqc-imgs/*.jpg)";
    string dst = R"(/media/ps/data/train/LQ/LQ/bdms/bdmask/workspace/inffff/inf)";

    vector<cv::String> files_;
    files_.reserve(10000);

    cv::glob(src, files_, true);
    vector<string> files(files_.begin(), files_.end());
    vector<float> avg_times1;

    assert(result1);
    int num_imp = 1;
    int noc = 1;
    while(noc){
        for(int im_idx=0; im_idx < files.size(); ++im_idx){
            cv::Mat image = cv::imread(files[im_idx], 0);
            boost::filesystem::path path(files[im_idx]);
            string nimp_result = dst + "/" + path.stem().string()  + ".jpg";
            auto begin_time1 = iLogger::timestamp_now_float();
            auto defect_res = bdmapp.bdmapp(fcos1,image);
            auto end_time1 = iLogger::timestamp_now_float();

            cv::cvtColor(image, image, cv::COLOR_GRAY2BGR);
            // 绘制到图片上
            int idx = 0;
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
                    cv::imwrite(nimp_result_mask,
                                box_mask);

                    cv::Mat edges;
                    std::vector<std::vector<cv::Point>> contours;
                    std::vector<cv::Vec4i> hierarhy;
                    cv::findContours(box_mask, contours, hierarhy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
                    cv::Point topleft(box.left, box.top);

                    for (auto& contour:contours){
                        for (auto& point:contour){
                            point += topleft;
                        }
                    }
                    cv::drawContours(image, contours, -1, cv::Scalar(0, 0, 255), 2);
                    idx++;
                    }
            }

            printf("***********第%d張圖片---当前%s, 图片有%d个缺陷, 当前推理耗时:%f***********\n", num_imp,
                        path.stem().string().c_str(), defect_res.size(), end_time1 - begin_time1);

            cv::imwrite(nimp_result, image);
            ++num_imp;

            avg_times1.emplace_back(end_time1 - begin_time1);
        }
        noc--;
    }
    float sum = 0;
    for (int i = 1; i < avg_times1.size(); i++) {
        sum += avg_times1[i];
    };

    printf("第一张卡共测试%d张, 总耗时:%.4f秒, 图片平均耗时: %.4f毫秒\n", avg_times1.size(), sum/1000, (sum / avg_times1.size()));
    printf("success\n");
    return 0;
}