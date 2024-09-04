// #include <iostream>
// #include <vector>
// #include <cuda_runtime.h>
// #include <common/cuda_tools.hpp>
// #include <common/trt_tensor.hpp>
// #include <opencv2/opencv.hpp>


// // Kernel function to find the maximum and minimum values in an array
// __global__ void findMinMaxKernel3(const float* data,  float* minResult,  float* maxResult,size_t n) {
//     extern __shared__ float sharedData[];

//     unsigned int tid = threadIdx.x;
//     unsigned int index = blockIdx.x * blockDim.x + tid;

//     // Load elements into shared memory
//     if (index < n) {
//         sharedData[tid] = data[index];                   // For max computation
//         sharedData[blockDim.x + tid] = data[index];      // For min computation
//     } else {
//         sharedData[tid] = -FLT_MAX;
//         sharedData[blockDim.x + tid] = FLT_MAX;
//     }
//     __syncthreads();

//     // Reduction to find the maximum and minimum
//     for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
//         if (tid < s && index + s < n) {
//             sharedData[tid] = fmaxf(sharedData[tid], sharedData[tid + s]);
//             sharedData[blockDim.x + tid] = fminf(sharedData[blockDim.x + tid], sharedData[blockDim.x + tid + s]);
//         }
//         __syncthreads();
//     }

//     // Write the result for this block to global memory
//     if (tid == 0) {
//         maxResult[blockIdx.x] = sharedData[0];
//         minResult[blockIdx.x] = sharedData[blockDim.x];
//     }
// }

// void findMinMax_device(float * data, float *d_maxResult, float * d_minResult, int size, int & grid_x, cudaStream_t stream){
//     // Allocate device memory

//     auto jobs = size;
//     auto grid = CUDATools::grid_dims(jobs);
//     auto block = CUDATools::block_dims(jobs);
//     grid_x = grid.x;
//     checkCudaKernel(findMinMaxKernel3<<<grid, block, block.x *sizeof(float)*2, stream>>>(data, d_minResult, d_maxResult, jobs));
//     auto min_val = std::min_element(d_minResult, d_minResult+512);
//     std::cout<<"内部最小值:"<<min_val << std::endl;
// }


// int main() {

//     // 初始化 CUDA
//     cudaStream_t stream_ = nullptr;
//     cudaSetDevice(0);
//     TRT::Tensor output_device(TRT::DataType::Float);
//     TRT::Tensor d_maxResult(TRT::DataType::Float);
//     TRT::Tensor d_minResult(TRT::DataType::Float);
//     output_device.resize(1, 1, 512, 512);
//     int output_device_size = output_device.size(0) * output_device.size(1) * output_device.size(2) * output_device.size(3);
//     d_maxResult.resize(1, (output_device_size +512 - 1 / 512));
//     d_minResult.resize(1, (output_device_size +512 - 1 / 512));
//     output_device.load_from_file(R"(/media/ps/data1/train/LQ/task/bdm/bdmask/workspace/models/cfa/inf/output_array_device)");
//     float * output_ptr = output_device.gpu<float>();
//     float * d_maxResult_ptr = d_maxResult.gpu<float>();
//     float * d_minResult_ptr = d_minResult.gpu<float>();
    

//     // float minVal;
//     // float maxVal; 
//     int grid_x;
//     // GPU 运行
//     findMinMax_device(output_ptr,  d_maxResult_ptr, d_minResult_ptr,512*512, grid_x, stream_);
//     float * d_maxResult_host = d_maxResult.cpu<float>();
//     float * d_minResult_host = d_minResult.cpu<float>();
//     auto minVal = std::min_element(d_minResult_host, d_minResult_host+grid_x);
//     auto maxVal = std::max_element(d_maxResult_host, d_maxResult_host+grid_x);
    

//     std::cout << minVal[0] << "  " << maxVal[0] << std::endl;

//     return 0;
// }



