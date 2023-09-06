#include <assert.h>
#include <algorithm>
#include <cstdio>
#include <boost/filesystem.hpp>


#include <common/ilogger.hpp>
#include <opencv2/opencv.hpp>

#include "bdmapp.h"
#include "app_bdm/bdm.hpp"


using namespace std;

static const char* cocolabels[] = {
     "1",  "2",  "3",  "4",  "5",  "6",  "7",  "8",  "9", "10",
    "11", "12", "13", "14", "15", "16", "17", "18", "19", "20",
    "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", 
    "31", "32", "33", "34", "35", "36", "37", "38", "39", "40", 
    "41", "42", "43", "44", "45", "46", "47", "48", "49", "50",
};

int main(){

    string fcos_engine_path = R"(/media/ps/data/train/LQ/LQ/bdmask4/workspace/OQC/model_0545999/model_0545999.trt)";
    string blend_engine_path = R"(/media/ps/data/train/LQ/LQ/bdmask4/workspace/OQC/boxfeat/model_blender-oqc-dy-boxfeat--fp16)";


    BdmApp bdmapp;
    shared_ptr<Bdm::Infer> bdm = nullptr;
    // shared_ptr<Bdm::Infer> bdm = nullptr;
    int device_id1 = 2;

    float mean[] = {59.406};
    float std[] = {59.32};



    bool result1 = bdmapp.bdminit(bdm, fcos_engine_path, blend_engine_path, mean, std, device_id1);
    
    //当前释放指针
    // bdm.reset();
    
    if(bdm != nullptr){
        string src = R"(/media/ps/data/train/LQ/LQ/bdmask4/workspace/mask_data/images/used/*.jpg)";
    
        // string src = R"(/media/ps/data/train/LQ/project/OQC/train/0001/go/train/*.jpg)";
        string dst = R"(/media/ps/data/train/LQ/LQ/bdmask4/workspace/mask_data/tinf)";

        vector<cv::String> files_;
        files_.reserve(10000);

        cv::glob(src, files_, true);
        vector<string> files(files_.begin(), files_.end());
        vector<float> avg_times1;

        assert(result1);
        int num_imp = 1;
        int noc = 10;
        while(noc){
            for(int im_idx=0; im_idx < files.size(); ++im_idx){
                cv::Mat image = cv::imread(files[im_idx], 0);
                boost::filesystem::path path(files[im_idx]);
                string nimp_result = dst + "/" + path.stem().string()  + ".jpg";
                string nimp_result_mask = dst + "/" + path.stem().string()  + "_mask.jpg";
                printf("当前推理图片:%s\n", path.stem().string().c_str());
                auto begin_time1 = iLogger::timestamp_now_float();
                auto defect_res = bdmapp.bdmapp(bdm, image);
                auto end_time1 = iLogger::timestamp_now_float();
                
                cv::cvtColor(image, image, cv::COLOR_GRAY2BGR);
                cv::Mat maskImage3C;
                double alpha = 0.8; // 控制合并的透明度
                cv::Mat combinedImage;
                cv::Mat image_mask = cv::Mat::zeros(2048, 2048, CV_8UC1);

                // // 绘制到图片上
                for(auto & box : defect_res){
                    cv::Scalar color(0, 255, 0);
                    cv::rectangle(image, cv::Point(box.left, box.top), cv::Point(box.right, box.bottom), color, 3);
                    auto name      = cocolabels[box.class_label];
                    auto caption   = cv::format("%s %.2f", name, sqrt(box.confidence));
                    int text_width = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;

                    // printf("%d-%f-%f-%f-%f-%f\n", box.class_label, box.left, box.top, box.right, box.bottom, sqrt(box.confidence));
                    cv::rectangle(image, cv::Point(box.left-3, box.top-33), cv::Point(box.left + text_width, box.top), color, -1);
                    cv::putText(image, caption, cv::Point(box.left, box.top-5), 0, 1, cv::Scalar::all(0), 2, 16);
                    cv::add(image_mask, box.mask, image_mask);
                }
                cv::cvtColor(image_mask, image_mask, cv::COLOR_GRAY2BGR);
                // cv::addWeighted(image, alpha, image_mask.clone(), 1 - alpha, 0, image);
            

                printf("***********第%d張圖片---当前%s, 图片有%d个缺陷, 当前推理耗时:%f***********\n", num_imp,
                            path.stem().string().c_str(), defect_res.size(), end_time1 - begin_time1);
                // // 保存结果图，并输出结果
                cv::imwrite(nimp_result, image);
                cv::imwrite(nimp_result_mask, image_mask);
                num_imp++;
                avg_times1.push_back(end_time1 - begin_time1);
            }
            noc--;
        }
        float sum = 0;
        for (int i = 1; i < avg_times1.size(); i++) {
            sum += avg_times1[i];
        };

        printf("第一张卡共测试%d张, 总耗时:%.4f秒, 图片平均耗时: %.4f毫秒\n", avg_times1.size(), sum/1000, (sum / (avg_times1.size()-1)));
        // printf("success\n");

    }else{
        printf("當前模型已釋放\n");
    }

    
    return 0;
}