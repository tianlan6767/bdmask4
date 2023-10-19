#include <assert.h>
#include <algorithm>
#include <cstdio>
#include <boost/filesystem.hpp>

#include <common/ilogger.hpp>
#include <opencv2/opencv.hpp>

#include "bdmapp.h"
#include "app_fcos/fcos.hpp"
#include "app_blender/blender.hpp"


using namespace std;

static const char* cocolabels[] = {
     "1",  "2",  "3",  "4",  "5",  "6",  "7",  "8",  "9", "10",
    "11", "12", "13", "14", "15", "16", "17", "18", "19", "20",
    "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", 
    "31", "32", "33", "34", "35", "36", "37", "38", "39", "40", 
    "41", "42", "43", "44", "45", "46", "47", "48", "49", "50",
};

int main(){
    // string fcos_engine_path = R"(/media/ps/data/train/LQ/LQ/bdmask4/workspace/Q1/model-FCOS-Q1-2.trtmode)";
    // string blend_engine_path = R"(/media/ps/data/train/LQ/LQ/bdmask4/workspace/Q1/model_blender-Q1.trtmodel)";

    string fcos_engine_path = R"(/media/ps/data/train/LQ/LQ/bdms/bdmask/workspace/models/JT/model_0826-dy)";
    string blend_engine_path = R"(/media/ps/data/train/LQ/LQ/bdms/bdmask/workspace/models/JT/blender-dy)";


    BdmApp bdmapp;
    shared_ptr<Fcos::Infer> fcos1 = nullptr;
    shared_ptr<Blender::Infer> blender1 = nullptr;

    int device_id1 = 0;

    float mean[] = {41,41,41};
    float std[] = {34,34,34};


    bool result1 = bdmapp.bdminit(fcos1, blender1, fcos_engine_path, blend_engine_path, mean, std, device_id1);

    string src = R"(/media/ps/data/train/LQ/LQ/bdms/bdmask/workspace/models/JT/imgs/*.jpg)";
    string dst = R"(/media/ps/data/train/LQ/LQ/bdms/bdmask/workspace/models/JT/inf333)";

    vector<cv::String> files_;
    files_.reserve(10000);

    cv::glob(src, files_, true);
    vector<string> files(files_.begin(), files_.end());
    vector<float> avg_times1;

    assert(result1);
    int num_imp = 0;
    int noc = 1;
    while(noc){
        for(int im_idx=0; im_idx < files.size(); ++im_idx){
            cv::Mat image = cv::imread(files[im_idx], 0);
            boost::filesystem::path path(files[im_idx]);
            string nimp_result = dst + "/" + path.stem().string()  + ".jpg";
            auto begin_time1 = iLogger::timestamp_now_float();
            auto defect_res = bdmapp.bdmapp(fcos1, blender1, image);
            auto end_time1 = iLogger::timestamp_now_float();

            cv::cvtColor(image, image, cv::COLOR_GRAY2BGR);
            // 绘制到图片上
            for(auto & box : defect_res.boxes){
                cv::Scalar color(0, 255, 0);
                cv::rectangle(image, cv::Point(box.left, box.top), cv::Point(box.right, box.bottom), color, 3);
                auto name      = cocolabels[box.class_label];
                auto caption   = cv::format("%s %.2f", name, sqrt(box.confidence));
                int text_width = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;

                // printf("%d-%f-%f-%f-%f-%f\n", box.class_label, box.left, box.top, box.right, box.bottom, sqrt(box.confidence));
                cv::rectangle(image, cv::Point(box.left-3, box.top-33), cv::Point(box.left + text_width, box.top), color, -1);
                cv::putText(image, caption, cv::Point(box.left, box.top-5), 0, 1, cv::Scalar::all(0), 2, 16);
            }

            printf("***********第%d張圖片---当前%s, 图片有%d个缺陷, 当前推理耗时:%f***********\n", num_imp,
                        path.stem().string().c_str(), defect_res.labels.size(), end_time1 - begin_time1);
            ++num_imp;

            int idx=0;
            for(auto& m :defect_res.masks){
                
                string nimp_result_mask = dst + "/" + path.stem().string() +"_mask_" + to_string(idx) +".jpg";
                cv::imwrite(nimp_result_mask, m);
                ++idx;
            }
            // 保存结果图，并输出结果
            cv::imwrite(nimp_result, image);
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