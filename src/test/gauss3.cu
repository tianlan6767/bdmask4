// #include <cuda_fp16.h>
// #include <cuda_runtime.h>
// #include <device_launch_parameters.h>
// #include <opencv2/opencv.hpp>
// #include <iostream>

// // 高斯核大小
// #define KERNEL_SIZE 33

// // 高斯核标准差
// #define SIGMA 4.0f

// // 定义高斯核函数
// __host__ __device__ float gaussian_kernel(int x, int y) {
//   return exp(-(x * x + y * y) / (2 * SIGMA * SIGMA));
// }

// // CUDA 内核函数
// __global__ void gaussian_filter_kernel(const float* input, float* output, int width, int height, const float* kernel) {
//   int x = blockIdx.x * blockDim.x + threadIdx.x;
//   int y = blockIdx.y * blockDim.y + threadIdx.y;

//   if (x >= width || y >= height) {
//     return;
//   }

//   float sum = 0.0f;
//   float weight_sum = 0.0f;

//   // 遍历高斯核
//   for (int i = -KERNEL_SIZE / 2; i <= KERNEL_SIZE / 2; ++i) {
//     for (int j = -KERNEL_SIZE / 2; j <= KERNEL_SIZE / 2; ++j) {
//       int dx = x + i;
//       int dy = y + j;

//       // 检查是否在图像边界内
//       if (dx >= 0 && dx < width && dy >= 0 && dy < height) {
//         int k_idx = (i + KERNEL_SIZE / 2) * KERNEL_SIZE + (j + KERNEL_SIZE / 2);
//         sum += static_cast<float>(input[dy * width + dx]) * kernel[k_idx];
//         weight_sum += kernel[k_idx];
//       }
//     }
//   }

//   output[y * width + x] = sum / weight_sum;
// }

// int main() {
//   // 加载输入图像
//   cv::Mat input_image = cv::imread("/media/ps/data1/train/LQ/task/bdm/bdmask/workspace/Pikachu.jpg", cv::IMREAD_GRAYSCALE);
//   if (input_image.empty()) {
//     std::cerr << "无法打开输入图像文件！" << std::endl;
//     return 1;
//   }

//   // 获取图像尺寸
//   int width = input_image.cols;
//   int height = input_image.rows;

//   // 预先计算高斯核
//   float kernel[KERNEL_SIZE * KERNEL_SIZE];
//   for (int i = -KERNEL_SIZE / 2; i <= KERNEL_SIZE / 2; ++i) {
//     for (int j = -KERNEL_SIZE / 2; j <= KERNEL_SIZE / 2; ++j) {
//       int k_idx = (i + KERNEL_SIZE / 2) * KERNEL_SIZE + (j + KERNEL_SIZE / 2);
//       kernel[k_idx] = gaussian_kernel(i, j);
//     }
//   }

//   // 分配主机内存
//   float* h_input = new float[width * height];
//   float* h_output = new float[width * height];

//   // 将 OpenCV 图像数据转换为 float 类型
//   for (int y = 0; y < height; ++y) {
//     for (int x = 0; x < width; ++x) {
//       h_input[y * width + x] = static_cast<float>(input_image.at<uchar>(y, x)) / 255.0f;
//     }
//   }

//   // 分配设备内存
//   float* d_input;
//   float* d_output;
//   float* d_kernel;
//   cudaMalloc(&d_input, width * height * sizeof(float));
//   cudaMalloc(&d_output, width * height * sizeof(float));
//   cudaMalloc(&d_kernel, KERNEL_SIZE * KERNEL_SIZE * sizeof(float));

//   // 将输入数据和高斯核复制到设备内存
//   cudaMemcpy(d_input, h_input, width * height * sizeof(float), cudaMemcpyHostToDevice);
//   cudaMemcpy(d_kernel, kernel, KERNEL_SIZE * KERNEL_SIZE * sizeof(float), cudaMemcpyHostToDevice);

//   // 设置线程块和网格尺寸
//   dim3 blockDim(16, 16);
//   dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

//   // 启动内核
//   gaussian_filter_kernel<<<gridDim, blockDim>>>(d_input, d_output, width, height, d_kernel);

//   // 将结果复制回主机内存
//   cudaMemcpy(h_output, d_output, width * height * sizeof(float), cudaMemcpyDeviceToHost);

//   // 将 float 类型数据转换为 OpenCV 图像数据
//   cv::Mat output_image(height, width, CV_8U);
//   for (int y = 0; y < height; ++y) {
//     for (int x = 0; x < width; ++x) {
//       output_image.at<uchar>(y, x) = static_cast<uchar>(h_output[y * width + x] * 255.0f);
//     }
//   }

//   // 保存结果图像
//   cv::imwrite("output333.jpg", output_image);

//   // 释放设备内存
//   cudaFree(d_input);
//   cudaFree(d_output);
//   cudaFree(d_kernel);

//   // 释放主机内存
//   delete[] h_input;
//   delete[] h_output;

//   return 0;
// }