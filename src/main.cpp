// #include <torch/script.h>
// #include <torch/torch.h>
// #include <trt_tensor.hpp>
// #include <unordered_map>
// #include <iostream>
// #include <vector>


// using namespace std;

// int main(){
//     // torch::Tensor output;
//     // cout << "cuda is_available:" << torch::cuda::is_available() << endl;
//     // torch::DeviceType device = at::kCPU;
//     // if(torch::cuda::is_available())
//     //     device = at::kCUDA;
//     // output = torch::randn({3, 3}).to(device);
//     // cout << output << endl;


//     TRT::Tensor box(TRT::DataType::Float);
//     TRT::Tensor base(TRT::DataType::Float);
//     TRT::Tensor feat(TRT::DataType::Float);

//     box.load_from_file("/media/ps/data/train/LQ/LQ/bdmask6/workspace/data/box_input");
//     base.load_from_file("/media/ps/data/train/LQ/LQ/bdmask6/workspace/data/base_input");
//     feat.load_from_file("/media/ps/data/train/LQ/LQ/bdmask6/workspace/data/top_feat_input");
//     auto box_cpu = box.cpu<float>();
//     auto base_cpu = base.cpu<float>();
//     auto feat_cpu = feat.cpu<float>();
//     auto box_tensor = torch::from_blob(box_cpu, {1, 5}, torch::kFloat32);
//     auto base_tensor = torch::from_blob(base_cpu, {1, 4, 512, 512}, torch::kFloat32);
//     auto feat_tensor = torch::from_blob(feat_cpu, {1, 784}, torch::kFloat32);
//     // auto size = box_tensor.sizes();
//     // printf("sizes: {%ld, %ld, %ld}\n", size[0], size[1]);

//     return 0;  
// }