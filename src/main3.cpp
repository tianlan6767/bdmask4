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

// void process_image(BdmApp& bdmapp, shared_ptr<Fcos::Infer> fcos, shared_ptr<Blender::Infer> blender, cv::Mat image, promise<long long> promise, int device_id) {

//     auto begin_time = std::chrono::high_resolution_clock::now();
//     auto defect_res = bdmapp.bdmapp(fcos, blender, image);
//     auto end_time = std::chrono::high_resolution_clock::now();
//     auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - begin_time);
//     printf("Device %d , %d defects, inference time: %lld ms\n", device_id, defect_res.labels.size(), duration.count());
//     // 将推理时间传递给 promise
//     promise.set_value(duration.count());
// }

// int main(){
//     string fcos_engine_path = R"(/media/ps/data/train/LQ/LQ/bdmask/workspace/used0/model-FCOS-OQC.trtmodel)";
//     string blend_engine_path = R"(/media/ps/data/train/LQ/LQ/bdmask/workspace/used0/model-blender.trtmodel)";



//     vector<cv::String> files_;
//     files_.reserve(10000);

//     string src = R"(/media/ps/data/train/LQ/project/OQC/test/0607/imgs/*.jpg)";
//     cv::glob(src, files_, true);
//     vector<string> files(files_.begin(), files_.end());

//     int device_id1 = 0;
//     int device_id2 = 1;
//     float mean[] = {59.406};
//     float std[] = {59.32};


//     // 读取所有图像并保存在 vector 中
//     std::vector<cv::Mat> images;
//     for (auto& file : files) {
//         cv::Mat image = cv::imread(file, 0);
//         if (!image.empty()) {
//             images.push_back(image);
//         }
//     }
    
//     BdmApp bdmapp;
//     shared_ptr<Fcos::Infer> fcos1 = nullptr;
//     shared_ptr<Blender::Infer> blender1 = nullptr;

//     shared_ptr<Fcos::Infer> fcos2 = nullptr;
//     shared_ptr<Blender::Infer> blender2 = nullptr;

    
//     bool result1 = bdmapp.bdminit(fcos1, blender1, fcos_engine_path, blend_engine_path, mean, std, device_id1);
//     bool result2 = bdmapp.bdminit(fcos2, blender2, fcos_engine_path, blend_engine_path, mean, std, device_id2);

//     // 创建两个线程，分别利用每张显卡进行推理
//     std::vector<std::thread> threads;
//     std::vector<std::future<long long>> futures;
//     threads.emplace_back([&] {
//         for (size_t i = 0; i < images.size(); i += 2) {
//             // 在第一张显卡上进行推理
//             std::promise<long long> promise;
//             futures.emplace_back(promise.get_future());
//             std::thread t(process_image, std::ref(bdmapp), fcos1, blender1, images[i], std::move(promise), device_id1);
//             t.detach();
//         }
//     });
//     threads.emplace_back([&] {
//         for (size_t i = 1; i < files.size(); i += 2) {
//             // 在第二张显卡上进行推理
//             std::promise<long long> promise;
//             futures.emplace_back(promise.get_future());
//             std::thread t(process_image, std::ref(bdmapp), fcos2, blender2, images[i], std::move(promise), device_id2);
//             t.detach();
//         }
//     });

//     // 等待所有任务完成，并输出推理时间
//     long long total_duration = 0;
//     for (auto& future : futures) {
//         total_duration += future.get();
//     }
//     std::cout << "Total inference time: " << total_duration << " ms" << std::endl;

//     // 等待所有线程完成
//     for (auto& thread : threads) {
//         thread.join();
//     }

//     return 0;
// }