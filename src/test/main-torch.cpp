// #include <atomic>
// #include <mutex>
// #include <queue>
// #include <condition_variable>
// #include <infer/trt_infer.hpp>
// #include <common/ilogger.hpp>
// #include <common/infer_controller.hpp>
// #include <common/preprocess_kernel.cuh>
// #include <common/monopoly_allocator.hpp>
// #include <common/cuda_tools.hpp>

// #include <torch/torch.h>
// #include <iostream>

// std::vector<torch::Tensor> splitBoxes(const torch::Tensor& boxes) {
//     std::vector<torch::Tensor> split_tensors = torch::split(boxes, 1, 1); // each is Nx1
//     return split_tensors;
// }

// torch::Tensor do_mask(const torch::Tensor& masks, const torch::Tensor& boxes) {
//     torch::Device device = masks.device();
//     auto splitBoxes_value = splitBoxes(boxes); // each is Nx1
//     torch::Tensor x0 = splitBoxes_value[0];
//     torch::Tensor y0 = splitBoxes_value[1];
//     torch::Tensor x1 = splitBoxes_value[2];
//     torch::Tensor y1 = splitBoxes_value[3];
    
//     torch::Tensor x0_int = x0.to(torch::kInt).squeeze();
//     torch::Tensor y0_int = y0.to(torch::kInt).squeeze();
//     torch::Tensor x1_int = x1.to(torch::kInt).squeeze();
//     torch::Tensor y1_int = y1.to(torch::kInt).squeeze();

//     int N = masks.size(0);

//     torch::Tensor img_y = torch::arange(y0_int.item<float>(), y1_int.item<float>(), torch::device(device).dtype(torch::kFloat32)) + 0.5f;
//     torch::Tensor img_x = torch::arange(x0_int.item<float>(), x1_int.item<float>(), torch::device(device).dtype(torch::kFloat32)) + 0.5f;
//     img_y = (img_y - y0) / (y1 - y0) * 2 - 1;
//     img_x = (img_x - x0) / (x1 - x0) * 2 - 1;
//     // img_x, img_y have shapes (N, w), (N, h)

//     torch::Tensor gx = img_x.unsqueeze(1).expand({N, img_y.size(1), img_x.size(1)});
//     torch::Tensor gy = img_y.unsqueeze(2).expand({N, img_y.size(1), img_x.size(1)});
//     torch::Tensor grid = torch::stack({gx, gy}, 3);
    
    // torch::Tensor img_masks = torch::grid_sampler_2d(masks, grid.to(masks.dtype()), 0, 0, false); // align_corners=false
    
//     return img_masks;
// }

// int main() {

//     // 指定要使用的 GPU 设备索引
//     int gpu_index = 2;
//     torch::Device device(torch::kCUDA, gpu_index);

//     // 创建输入tensor
//     TRT::Tensor output_mask_pred(TRT::DataType::Float16);
//     output_mask_pred.load_from_file("/media/ps/data/train/LQ/LQ/bdms/bdmask/workspace/models/JT/inf/output-nogrid-sampler");
//     float* data = output_mask_pred.cpu<float>();
//     // 指定形状
//     std::vector<int64_t> shape = {1, 1, 56, 56};
//     torch::Tensor input = torch::from_blob(data, shape, torch::kFloat32);
//     // 创建采样坐标tensor
//     torch::Tensor boxes = torch::tensor({784.1294, 293.5286, 898.5332, 411.1134}).unsqueeze(0);

//     std::cout << "Shape of boxes: " << boxes.sizes() << std::endl;
//     auto masks = do_mask(input, boxes).squeeze();
//     std::cout << "Shape of boxes: " << masks.sizes() << std::endl;

//     // 将 masks 转换为 NumPy 数组
//     cv::Mat masks_mat(masks.size(0), masks.size(1), CV_32F, masks.data_ptr<float>());

//     // 使用 np.where() 进行阈值处理
//     cv::Mat thresholded_masks;
//     cv::threshold(masks_mat, thresholded_masks, 0.5, 255, cv::THRESH_BINARY);

//     // 保存为图像文件
//     std::string output_path = "/media/ps/data/train/LQ/LQ/bdms/bdmask/workspace/models/JT/inf/111/output.png";
//     cv::imwrite(output_path, thresholded_masks);
//     // torch::Tensor output = torch::grid_sampler(input, grid, 0,0, false);
//     return 0;
// }