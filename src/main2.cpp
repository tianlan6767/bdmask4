// #include "app_fcos/fcos.hpp"
// #include "app_blender/blender.hpp"
// #include <common/ilogger.hpp>
// #include <opencv2/opencv.hpp>
// #include <algorithm>
// #include <cstdio>
// #include "bdmapp.h"
// #include <thread>
// #include <future>
// #include <chrono>

// using namespace std;

// void process_image(BdmApp& bdmapp, shared_ptr<Fcos::Infer> fcos, shared_ptr<Blender::Infer> blender, cv::Mat image, promise<long long> promise, int device_id, int& count, long long & total_time) {

//     auto begin_time = std::chrono::high_resolution_clock::now();
//     auto defect_res = bdmapp.bdmapp(fcos, blender, image);
//     auto end_time = std::chrono::high_resolution_clock::now();
//     auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - begin_time);
//     printf("Device %d processed image %d, has %d defects, inference time: %lld ms\n", device_id, count, defect_res.labels.size(), duration.count());
//     count++;
//     total_time += duration.count();
//     promise.set_value(duration.count());
//     promise.set_value(total_time);
// }

// int main(){
//     string fcos_engine_path = R"(/media/ps/data/train/LQ/LQ/bdmask/workspace/used0/model-FCOS-OQC.trtmodel)";
//     string blend_engine_path = R"(/media/ps/data/train/LQ/LQ/bdmask/workspace/used0/model-blender.trtmodel)";
//     BdmApp bdmapp;
//     shared_ptr<Fcos::Infer> fcos1 = nullptr;
//     shared_ptr<Blender::Infer> blender1 = nullptr;

//     shared_ptr<Fcos::Infer> fcos2 = nullptr;
//     shared_ptr<Blender::Infer> blender2 = nullptr;

//     int device_id1 = 0;
//     int device_id2 = 1;
//     float mean[] = {59.406};
//     float std[] = {59.32};
//     bool result1 = bdmapp.bdminit(fcos1, blender1, fcos_engine_path, blend_engine_path, mean, std, device_id1);
//     bool result2 = bdmapp.bdminit(fcos2, blender2, fcos_engine_path, blend_engine_path, mean, std, device_id2);
//     string src = R"(/media/ps/data/train/LQ/project/OQC/test/0607/imgs/*.jpg)";
//     vector<cv::String> files_;
//     files_.reserve(10000);

//     cv::glob(src, files_, true);
//     vector<string> files(files_.begin(), files_.end());

//     assert (result1 && result2);

//     // 创建线程池
//     vector<future<long long>> futures;
//     int count1 = 0;
//     int count2 = 0;
//     long long total_time1 = 0;
//     long long total_time2 = 0;
//     printf("文件夹图像数量 %d\n", files.size());


//     long long total_time0 = 0;
//     for (int im_idx = 0; im_idx < files.size(); ++im_idx) {
//         cv::Mat image = cv::imread(files[im_idx], 0);
//         auto begin_time0 = std::chrono::high_resolution_clock::now();
//         if (im_idx % 2 == 0) {
//             promise<long long> p;
//             futures.emplace_back(p.get_future());
//             async(launch::async, process_image, ref(bdmapp), fcos1, blender1, image, move(p), device_id1, ref(count1), ref(total_time1));
//         } else {
//             promise<long long> p;
//             futures.emplace_back(p.get_future());
//             async(launch::async, process_image, ref(bdmapp), fcos2, blender2, image, move(p), device_id2, ref(count2), ref(total_time2));
//         }
//         auto end_time0 = std::chrono::high_resolution_clock::now();
//         auto duration0 = std::chrono::duration_cast<std::chrono::milliseconds>(end_time0 - begin_time0);
//         auto all_seq_time = duration0.count();
//         total_time0 += all_seq_time;
//     }

//     // 等待所有线程完成
//     auto begin_time1 = std::chrono::high_resolution_clock::now();
//     long long total_time = 0;
//     for (auto& future : futures) {
//         total_time += future.get();
//     }

//     auto end_time1 = std::chrono::high_resolution_clock::now();
//     auto duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(end_time1 - begin_time1);
//     auto all_total_time = duration1.count();
//     printf("Overall total inference : %lld ms\n", all_total_time);
//     printf("Overall total inference2 : %lld ms\n", total_time0);

//     // printf("Overall inference time: %.2f ms\n", static_cast<double>(total_time));
//     // Compute and print average inference time for each device
//     double average_time1 = static_cast<double>(total_time1) / count1;
//     double average_time2 = static_cast<double>(total_time2) / count2;
//     printf("显卡0 processed %d images in %lld ms (average: %.2f ms per image)\n", count1, total_time1, average_time1);
//     printf("显卡1 processed %d images in %lld ms (average: %.2f ms per image)\n", count2, total_time2, average_time2);

//     // Compute and print overall average inference time
//     // double overall_average_time = static_cast<double>(total_time) / files.size();
    
//     return 0;
// }