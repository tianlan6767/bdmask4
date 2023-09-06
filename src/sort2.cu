// #include <stdio.h>
// #include <math.h>
// #include <cuda.h>
// #include <cuda_runtime.h>
// #include <time.h>


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

// using namespace std;

// const int N = 1024; // 输入数组的大小
// const int k = 50; // 要求的 topk 值的数量
// const int BLOCK_SIZE = 512; // 线程块大小

// __global__ void gpu_topk(float* input, float* output, int* output_idx, int N, int k) {
//     // 在共享内存中进行计算和排序
//     __shared__ float s_data[BLOCK_SIZE];

//     int tid = threadIdx.x;
//     int block_id = blockIdx.x;
//     int idx = tid + block_id * blockDim.x;

//     if (idx < N) {
//         // 将数据从全局内存拷贝到共享内存中
//         float* pbox = input + 1 + idx *  791;
//         int keepflag = pbox[6];
//         if (keepflag){
//             s_data[tid] = pbox[4];
//         }

//         // s_data[tid] = input[idx];
//     } else {
//         s_data[tid] = -1; // 如果数据不存在，则设置为一个极小值
//     }

//     __syncthreads();

//     // 进行并行排序
//     for (int i = 1; i < blockDim.x; i *= 2) {
//         int j = tid - i;
//         if (j >= 0 && s_data[j] < s_data[tid]) {
//             float tmp = s_data[j];
//             s_data[j] = s_data[tid];
//             s_data[tid] = tmp;
//         }
//         __syncthreads();
//     }

//     // 将排好序的数据和索引值写回全局内存
//     if (tid < k) {
//         output[block_id * k + tid] = s_data[tid];
//         for (int i = 0; i < N; i++) {
//             float* pbox = input + 1 + idx *  791;
//             int keepflag = pbox[6];
//             if (keepflag && (pbox[4] == s_data[tid])){
//                 output_idx[block_id * k + tid] = i;
//                 break;
//             }

//         }
//     }
// }

// int main() {
//     float input[N];
//     for (int i = 0; i < N; i++) {
//         input[i] = rand() / (float)RAND_MAX; // 随机生成输入数据
//     }

//     TRT::Tensor output_array_device(TRT::DataType::Float);
//     output_array_device.load_from_file("/media/ps/data/train/LQ/LQ/bdmask4/workspace/mask_data/output_array_device_");
//     float * output_array_ptr = output_array_device.gpu<float>();

//     float* gpu_input = nullptr;
//     float* gpu_output = nullptr;
//     int* gpu_output_idx = nullptr;

//     // 在 Device 上分配内存
//     cudaMalloc((void**)&gpu_input, N * sizeof(float));
//     cudaMalloc((void**)&gpu_output, k * sizeof(float));
//     cudaMalloc((void**)&gpu_output_idx, k * sizeof(int));

//     cudaEvent_t start, stop_gpu;
//     cudaEventCreate(&start);
//     cudaEventCreate(&stop_gpu);

//     // 将输入数据拷贝到 Device 上
//     cudaMemcpy(gpu_input, input, N * sizeof(float), cudaMemcpyHostToDevice);

//     // 计算 topk 值
//     int cr = 50;
//     cudaEventRecord(start);
//     cudaEventSynchronize(start);
//     for (int i = 0; i < cr; i++){
//         gpu_topk<<<1, BLOCK_SIZE>>>(output_array_ptr, gpu_output, gpu_output_idx, N, k);
//     }
//     cudaEventRecord(stop_gpu);
//     cudaEventSynchronize(stop_gpu);
//     float  time_gpu;
//     cudaEventElapsedTime(&time_gpu, start, stop_gpu);

    

//     // 将计算结果拷贝回 Host 上
//     float output[k];
//     int output_idx[k];
//     cudaMemcpy(output, gpu_output, k * sizeof(float), cudaMemcpyDeviceToHost);
//     cudaMemcpy(output_idx, gpu_output_idx, k * sizeof(int), cudaMemcpyDeviceToHost);

//     // 输出结果
//     std::cout << "Top " << k << " values:" << std::endl;
//     for (int i = 0; i < k; i++) {
//         std::cout << output[i] << " ";
//     }
//     std::cout << std::endl;

//     std::cout << "Top " << k << " indexes:" << std::endl;
//     for (int i = 0; i < k; i++) {
//         std::cout << output_idx[i] << " ";
//     }
//     std::cout << std::endl;
//     printf("GPU time: %.5f\n", (time_gpu / cr));

//     // 释放 Device 上的内存
//     cudaFree(gpu_input);
//     cudaFree(gpu_output);
//     cudaFree(gpu_output_idx);

//     return 0;
// }