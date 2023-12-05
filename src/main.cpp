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

#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>

using namespace std;
using json = nlohmann::json;

static const char* cocolabels[] = {
     "1",  "2",  "3",  "4",  "5",  "6",  "7",  "8",  "9", "10",
    "11", "12", "13", "14", "15", "16", "17", "18", "19", "20",
    "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", 
    "31", "32", "33", "34", "35", "36", "37", "38", "39", "40", 
    "41", "42", "43", "44", "45", "46", "47", "48", "49", "50",
};

void save_json(string save_dir, json resjson, string name){
    string json_file = save_dir + "/" + name + ".json"; 
    std::ofstream file(json_file);
    if (file.is_open()) {
        file << resjson.dump(4);
        file.close();
        std::cout << "JSON data saved to file." << std::endl;
    } else {
        std::cerr << "Unable to open file." << std::endl;
    }
}

void sum_times(vector<float> avg_times, int batch_size, vector<string> files, int noc, string name){
    float sum_mul = 0;
    for(int i = 0; i < avg_times.size(); i++) {
        sum_mul += avg_times[i];
    };
    printf("%s-共推理%d批, 共测试 %d 张, batch总耗时:%.4f秒, 图片平均耗时: %.4f毫秒\n", name.c_str(), avg_times.size(), files.size() * noc, sum_mul/1000, (sum_mul / (files.size()*noc)));
}

void batchsizes_pref(shared_ptr<Fcos::Infer> fcos, vector<string> &files, int batch_size, string save_dir, int noc, vector<float> &avg_times, json &resjson){
    int num_imp = 1;
    int num_noc = noc;
    // 多图推理
    while(noc){
        for(int im_idx= 0; im_idx < files.size(); im_idx += batch_size){
            vector<cv::Mat> images;
            vector<string> imns;
            for(int n = 0; n < batch_size; n++){
                if (im_idx + n >= files.size()) break;
                cv::Mat image = cv::imread(files[im_idx + n], 0);
                boost::filesystem::path path(files[im_idx + n]);
                string imn = path.stem().string() + ".jpg";
                string nimp_result = save_dir + "/" + imn;
                images.push_back(image);
                imns.push_back(imn);
            }
            auto begin_time1_mul = iLogger::timestamp_now_float();
            auto images_boxes = fcos->commits(images);
            images_boxes.back().get();
            auto end_time1_mul = iLogger::timestamp_now_float();
            avg_times.emplace_back(end_time1_mul - begin_time1_mul);
            for(int image_id= 0; image_id < images_boxes.size(); ++image_id){
                json regionsarray = json::array();
                auto imn = imns[image_id];
                auto image = images[image_id].clone();
                auto boxes= images_boxes[image_id].get();
                int box_idx = 0;
                if (image.channels()==1){
                    cv::cvtColor(image, image, cv::COLOR_GRAY2BGR);
                }
                for(auto & box : boxes){
                    cv::Scalar color(0, 255, 0);
                    cv::rectangle(image, cv::Point(box.left, box.top), cv::Point(box.right, box.bottom), color, 3);
                    auto name      = cocolabels[box.class_label];
                    auto caption   = cv::format("%s %.2f", name, sqrt(box.confidence));
                    int text_width = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
    
                    printf("%d  %f  %f  %f  %f  %f\n", box.class_label, box.left, box.top, box.right, box.bottom, sqrt(box.confidence));
                    cv::rectangle(image, cv::Point(box.left-3, box.top-33), cv::Point(box.left + text_width, box.top), color, -1);
                    cv::putText(image, caption, cv::Point(box.left, box.top-5), 0, 1, cv::Scalar::all(0), 2, 16);

                    if (box.seg) {
                        string nimp_result_mask = save_dir + "/" + imn +"_mask_" + to_string(box_idx) +".jpg";
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
                        ++box_idx;
                        }
                }
                resjson[imn] = {
                        {"filename", imn},
                        {"regions", regionsarray},
                        {"type", "inf"}
                };
                
                string nimp_result = save_dir + "/" + imn  + ".jpg";
                printf("***********第%d張圖片---当前%s, 图片有%d个缺陷, 当前推理 %d 张图片耗时平均:%.4f-%.4f***********\n", num_imp,
                            imn.c_str(), boxes.size(), images_boxes.size(), (end_time1_mul - begin_time1_mul), ((end_time1_mul - begin_time1_mul)/images_boxes.size()));
                cv::imwrite(nimp_result, image);
                ++num_imp;
                }
            }
        noc--;
    }
}



void batch_pref(shared_ptr<Fcos::Infer> fcos, vector<string> &files, string save_dir, int noc, vector<float> &avg_times, json &resjson){
    // 单图推理
    int num_imp = 1;
    int num_noc = noc;
    while(noc){
        for(int im_idx=0; im_idx < files.size(); ++im_idx){
            cv::Mat image = cv::imread(files[im_idx], 0);
            boost::filesystem::path path(files[im_idx]);
            string imn = path.stem().string() + ".jpg";
            string nimp_result = save_dir + "/" + path.stem().string()  + ".jpg";
            auto begin_time1 = iLogger::timestamp_now_float();
            auto defect_res = fcos->commit(image).get();
            auto end_time1 = iLogger::timestamp_now_float();
            avg_times.emplace_back(end_time1 - begin_time1);
            if (image.channels()==1){
                    cv::cvtColor(image, image, cv::COLOR_GRAY2BGR);
             }
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
                    string nimp_result_mask = save_dir + "/" + path.stem().string() +"_mask_" + to_string(idx) +".jpg";
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
                        imn.c_str(), defect_res.size(), end_time1 - begin_time1);

            cv::imwrite(nimp_result, image);
            ++num_imp;
        }
        noc--;
    }
}


void batch_pref_only(shared_ptr<Fcos::Infer> &fcos, vector<string> &files, int noc){
    // 单图推理
    while(noc){
        for(int im_idx=0; im_idx < files.size(); ++im_idx){
            cv::Mat image = cv::imread(files[im_idx], 0);
            auto defect_res = fcos->commit(image).get();
            for(auto & box : defect_res){
                printf("%d  %f  %f  %f  %f  %f\n", box.class_label, box.left, box.top, box.right, box.bottom, sqrt(box.confidence));
            } 
        }
        noc--;
    }
}


int main(){
    string fcos_engine_path = R"(/media/ps/data/train/LQ/task/bdm/bdmask/workspace/models/OQC/model_0364999-dy-b20-opt1)";
    BdmApp bdmapp;
    shared_ptr<Fcos::Infer> fcos1 = nullptr;
    shared_ptr<Fcos::Infer> fcos2 = nullptr;
    int device_id1 = 2;
    int device_id2 = 3;
    // float mean[] = {57.14, 55.92, 56.19};
    // float std[] = {61.46, 61.27, 61.23};

    float mean[] = {90};
    float std[] = {77};



    string src = R"(/media/ps/data/train/LQ/task/bdm/bdmask/workspace/code/ptq-bdm/data/data2/val/*.jpg)";
    string dst = R"(/media/ps/data/train/LQ/task/bdm/bdmask/workspace/models/OQC/tinf-b12)";
    iLogger::rmtree(dst);
    iLogger::mkdir(dst);
    vector<cv::String> files_;
    files_.reserve(10000);
    
    cv::glob(src, files_, true);
    vector<string> files(files_.begin(), files_.end());

    std::vector<std::string> files1;
    std::vector<std::string> files2;

    // 计算要分割的位置
    std::size_t splitIndex = files.size() / 2;

    // 使用迭代器将元素分割成两个子容器
    std::copy(files.begin(), files.begin() + splitIndex, std::back_inserter(files1));
    std::copy(files.begin() + splitIndex, files.end(), std::back_inserter(files2));

    // // warm-up
    // for(int i = 0; i < 5; ++i){
    //     cv::Mat image(2048, 2048, CV_8UC1);
    //     if(sizeof(mean) / sizeof(float) == 3){
    //         cv::cvtColor(image, image, cv::COLOR_GRAY2BGR);
    //     }
    //     image.setTo(128);
    //     fcos1->commit(image);
    //     fcos2->commit(image);
    // }
    
    // assert(result1);
    int noc_num = 1;
    // // 多图推理
    // json resjson_mul;
    // int batch_size_mul = 5;
    // vector<float> avg_times_mul;
    // batchsizes_pref(fcos1, files, batch_size_mul, dst, noc_num, avg_times_mul, resjson_mul);
    // save_json(dst, resjson_mul, "resjson_mul");
    
    // // 单图推理
    // json resjson_single;
    // int batch_size = 1;
    // vector<float> avg_times_single;
    // batch_pref(fcos2, files, dst, noc_num, avg_times_single, resjson_single);
    // save_json(dst, resjson_single, "resjson_single");

    // 多线程推理
    vector<thread> threads;
    vector<float> avg_times1;
    vector<float> avg_times2;
    json resjson1;
    json resjson2;
    bool result1 = bdmapp.bdminit(fcos1, fcos_engine_path, mean, std, device_id1);
    bool result2 = bdmapp.bdminit(fcos2, fcos_engine_path, mean, std, device_id2);
    auto threads_st = iLogger::timestamp_now_float();
    threads.emplace_back(batch_pref_only, std::ref(fcos1), std::ref(files1), noc_num);
    threads.emplace_back(batch_pref_only, std::ref(fcos2), std::ref(files2), noc_num);
    // threads.emplace_back(batch_pref, std::ref(fcos1), std::ref(files1), dst, noc_num, std::ref(avg_times1), std::ref(resjson1));
    // threads.emplace_back(batch_pref, std::ref(fcos2), std::ref(files2), dst, noc_num, std::ref(avg_times2), std::ref(resjson2));
    for (auto & thread: threads){
        thread.join();
    }
    auto threads_et = iLogger::timestamp_now_float();
    printf("多线程推理耗时:%.04f \n", (threads_et - threads_st)/1000);

    // 多进程推理
    // int batch_size = 1;
    // auto pid_st = iLogger::timestamp_now_float();
    // // Fork the first process for fcos1
    // pid_t pid1 = fork();
    // if (pid1 == -1){
    //     std::cerr << "failed to fork process for fcos1" << std::endl;
    //     return 1;
    // }else if(pid1 == 0){
    //     //child process for fcos1
    //     // Close the read end of the pipe
    //     bool result1 = bdmapp.bdminit(fcos1, fcos_engine_path, mean, std, device_id1);
    //     assert(result1);
    //     batch_pref_only(fcos1, files1, noc_num);
    //     // sum_times(avg_times_single, batch_size, files, noc_num, "单batch");
    //     exit(0);
    // }
    // pid_t pid2 = fork();
    // if (pid2 == -1){
    //     std::cerr << "failed to fork process for fcos1" << std::endl;
    //     return 1;
    // }else if(pid2 == 0){
    //     //child process for fcos1
    //     // Close the read end of the pipe
    //     bool result2 = bdmapp.bdminit(fcos2, fcos_engine_path, mean, std, device_id2);
    //     assert(result2);
    //     batch_pref_only(fcos2, files2, noc_num);
    //     exit(0);
    // }

    // // 等待两个子进程完成
    // int status;
    // waitpid(pid1, &status, 0);
    // waitpid(pid2, &status, 0);
    // auto pid_et = iLogger::timestamp_now_float();
    // printf("多进程推理耗时:%.04f \n", (pid_et - pid_st) / 1000);

    printf("success\n");
    return 0;
}