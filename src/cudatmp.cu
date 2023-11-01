// #include <iostream>
// #include <vector>
// #include <cuda_runtime.h>
// #include <common/cuda_tools.hpp>
// #include <common/trt_tensor.hpp>
// #include <opencv2/opencv.hpp>

// __global__ void resize_kernel(uint8_t* src, int src_width, int src_height, float* dst, int dst_width, int dst_height, int channels, int edge){

//         int position = blockDim.x * blockIdx.x + threadIdx.x;
//         if (position >= edge) return;

//         int dx      = position % dst_width;
//         int dy      = position / dst_width;
//         float c0, c1, c2;
//         int dst_area = dst_width * dst_height;
//         if (dx < dst_width && dy < dst_height) {
//             if (dx < src_width && dy < src_height) {
//                 int srcIdx_c0 = (dy * src_width + dx) * channels + 0;
//                 int srcIdx_c1 = (dy * src_width + dx) * channels + 1;
//                 int srcIdx_c2 = (dy * src_width + dx) * channels + 2;
//                 c0 = src[srcIdx_c0];
//                 c1 = src[srcIdx_c1];
//                 c2 = src[srcIdx_c2];
//             } else {
//                 c0 = 0.0f;
//                 c1 = 0.0f;
//                 c2 = 0.0f;
//             }
//             int dstIdx_c0 = dy * dst_width + dx;
//             int dstIdx_c1 = dstIdx_c0 + dst_area;
//             int dstIdx_c2 = dstIdx_c1 + dst_area;
//             dst[dstIdx_c0] = c0;
//             dst[dstIdx_c1] = c1;
//             dst[dstIdx_c2] = c2;
//     }
// }


// void resize_plane(
// 		uint8_t* src, int src_width, int src_height, float* dst, int dst_width, int dst_height,int channels, 
// 		cudaStream_t stream) {
		
// 		int jobs   = dst_width * dst_height;
// 		auto grid  = CUDATools::grid_dims(jobs);
// 		auto block = CUDATools::block_dims(jobs);

//         checkCudaKernel(resize_kernel << <grid, block, 0, stream >> > (src,src_width, src_height, dst,dst_width, dst_height, channels, jobs));		
// 	}


// int main() {

//     // 初始化 CUD
//     cudaStream_t stream_ = nullptr;
//     cudaSetDevice(3);
//     // 定义时间戳变量
//     cudaEvent_t start, stop;
//     cudaEventCreate(&start);
//     cudaEventCreate(&stop);
//     TRT::Tensor src_device(TRT::DataType::UInt8);
//     TRT::Tensor dst_device(TRT::DataType::Float);
//     cv::Mat image = cv::imread("/media/ps/data/train/LQ/LQ/bdms/bdmask/workspace/models/JT/JT-imgs/2222/odd.jpg", 1);
//     size_t size_image = image.cols * image.rows * image.channels();
//     src_device.resize(image.rows, image.cols, 3);
//     int output_width_ = 2144;
//     int output_height_ = 3584;
//     dst_device.resize(3, output_height_,output_width_);
//     uint8_t* src_gpu = src_device.gpu<uint8_t>();
//     float* dst_gpu = dst_device.gpu<float>();
//     checkCudaRuntime(cudaMemcpyAsync(src_gpu, image.data, size_image, cudaMemcpyHostToDevice, stream_));
//     resize_plane(
//         src_gpu,  image.cols, image.rows, 
//         dst_gpu, output_width_, output_height_, image.channels(), stream_);
        
//     // 启动计时
//     dst_device.save_to_file("/media/ps/data/train/LQ/LQ/bdms/bdmask/workspace/models/JT/JT-imgs/2222/dst");
//     cudaEventRecord(start, stream_);
    
//     // 停止计时
//     cudaEventRecord(stop, stream_);
//     cudaEventSynchronize(stop);

//     // 计算执行时间
//     float elapsedTime;
//     cudaEventElapsedTime(&elapsedTime, start, stop);

//     // 打印执行时间
//     std::cout << "CUDA Kernel Execution Time: " << elapsedTime << " ms" << std::endl;

//     // 释放时间戳资源
//     cudaEventDestroy(start);
//     cudaEventDestroy(stop);
    
//     return 0;
// }