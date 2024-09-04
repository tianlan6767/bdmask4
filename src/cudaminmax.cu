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

// // Function to find min and max in an array
// void findMinMax_cpu(const float* data, size_t n, float& min_val, float& max_val) {
//     min_val = data[0];
//     max_val = data[1];

//     for (size_t i = 0; i < n; ++i) {
//         if (data[i] < min_val) {
//             min_val = data[i];
//         }
//         if (data[i] > max_val) {
//             max_val = data[i];
//         }
//     }
// }

// // Kernel function to find the maximum value in an array
// __global__ void findMinMaxKernel2(const float *data,float *minResult,  float *maxResult, size_t n) {
//     extern __shared__ float sharedData[];

//     // Calculate thread ID
//     unsigned int tid = threadIdx.x;
//     unsigned int index = blockIdx.x * blockDim.x + tid;

//     // Load elements into shared memory
//     if (index < n) {
//         sharedData[tid] = data[index];
//     } else {
//         sharedData[tid] = 0.0f;
//     }
//     __syncthreads();

//     // Reduction to find the maximum and minimum
//     for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
//         if (tid < s && index + s < n) {
//             sharedData[tid] = fmaxf(sharedData[tid], sharedData[tid + s]);
//             sharedData[tid + blockDim.x] = fminf(sharedData[tid + blockDim.x], sharedData[tid + s + blockDim.x]);
//         }
//         __syncthreads();
//     }

//     // Write the result for this block to global memory
//     if (tid == 0) {
//         maxResult[blockIdx.x] = sharedData[0];
//         minResult[blockIdx.x] = sharedData[blockDim.x];
//     }
// }


// // 在GPU上计算最大值和最小值
// __global__ void findMinMaxKernel(const float* data, float* d_min, float* d_max, int n) {
//     extern __shared__ float sdata[];
//     int tid = threadIdx.x;
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     sdata[tid] = (i < n) ? data[i] : (tid < blockDim.x / 2 ? FLT_MAX : FLT_MIN);
//     __syncthreads();

//     for (unsigned int s = blockDim.x / 2; s > 0; s /= 2) {
//         if (tid < s) {
//             sdata[tid] = fmax(sdata[tid], sdata[tid + s]);
//             sdata[tid + blockDim.x / 2] = fmin(sdata[tid + blockDim.x / 2], sdata[tid + s + blockDim.x / 2]);
//         }
//         __syncthreads();
//     }

//     if (tid == 0) {
//         atomicMax((unsigned int*)d_max, __float_as_int(sdata[0]));
//         atomicMin((unsigned int*)d_min, __float_as_int(sdata[blockDim.x / 2]));
//     }
// }


// void findMinMax_device(float * data, float*minVal, float* maxVal, int size, cudaStream_t stream){
//     // Allocate device memory

//     auto jobs = size;
//     auto grid = CUDATools::grid_dims(jobs);
//     auto block = CUDATools::block_dims(jobs);

//     float *d_maxResult, *d_minResult;

//     cudaMalloc((void **)&d_maxResult, grid.x * sizeof(float));
//     cudaMalloc((void **)&d_minResult, grid.x * sizeof(float));
//     // checkCudaKernel(findMinMaxKernel<<<grid, block, block.x *sizeof(float), stream>>>(data, minVal, maxVal, jobs));
//     checkCudaKernel(findMinMaxKernel3<<<grid, block, block.x *sizeof(float)*2, stream>>>(data, d_minResult, d_maxResult, jobs));

//      // Copy results back to host
//     float *h_maxResult = new float[grid.x];
//     float *h_minResult = new float[grid.x];
//     cudaMemcpy(h_maxResult, d_maxResult, grid.x * sizeof(float), cudaMemcpyDeviceToHost);
//     cudaMemcpy(h_minResult, d_minResult, grid.x * sizeof(float), cudaMemcpyDeviceToHost);

//     // Perform final reduction on the host
//     float maxVal1 = h_maxResult[0];
//     float minVal2 = h_minResult[0];
//     for (size_t i = 1; i < grid.x; ++i) {
//         maxVal1 = fmaxf(maxVal1, h_maxResult[i]);
//         minVal2 = fminf(minVal2, h_minResult[i]);
//     }
//     std::cout << "Max value: " << maxVal1 << std::endl;
//     std::cout << "Min value: " << minVal2 << std::endl;
// }


// int main() {

//     // 初始化 CUDA
//     cudaStream_t stream_ = nullptr;
//     cudaSetDevice(0);
//     TRT::Tensor output_device(TRT::DataType::Float);
//     output_device.resize(1, 1, 512, 512);
//     output_device.load_from_file(R"(/media/ps/data1/train/LQ/task/bdm/bdmask/workspace/models/cfa/inf/output_array_device)");
//     float * output_ptr = output_device.gpu<float>();
    

//     float* d_max;
//     float *d_min; 
//     // 在 GPU 上分配内存并复制初始值
//     cudaMalloc(&d_max, sizeof(float));
//     cudaMalloc(&d_min, sizeof(float));

//     // GPU 运行
//     findMinMax_device(output_ptr, d_min, d_max,  512*512, stream_);

//     float * output_host = output_device.cpu<float>();
//     float min_val, max_val;
//     // Find min and max using standard library functions
//     findMinMax_cpu(output_host, 512*512, min_val, max_val);
    

//     float max_value, min_value;
//     // 将结果从 GPU 拷贝回主机端
//     cudaMemcpy(&max_value, d_max, sizeof(float), cudaMemcpyDeviceToHost);
//     cudaMemcpy(&min_value, d_min, sizeof(float), cudaMemcpyDeviceToHost);

//     cudaFree(d_max);
//     cudaFree(d_min);

//     std::cout << min_val  <<"  "<< max_val << std::endl;
    
//     return 0;
// }



