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
// // #include <app_fcos/fcos_decode.cu>

// using namespace std;

// #define N 20
// #define BLOCK_SIZE 64
// // #define BLOCK_SIZE 256
// #define GRID_SIZE 16
// #define topk 5

// __managed__ float source[N];
// __managed__ float gpu_result[topk];
// __managed__ int gpu_result_idxes[topk];
// __managed__ float _1_pass_result[topk * GRID_SIZE];
// __managed__ int _1_pass_result_idxes[topk * GRID_SIZE];

// // topK == 20
// // source[N]:  1 + 2 + 3 + 4 + ...............N
// // cpu: for loop
// // gpu: 1 + 2 + 3 + 4 + ...............N    0 + 1 + 2 + 3 + 4[20] + 5 + 6 + 7
// // thread id step 0:  tid0:source[0][20] > source[4][20]? source[0] & source[4]-> source[0][20]
// //                    tid1:source[1] + source[5] -> source[1]
// //                    tid2:source[2] + source[6] -> source[2]
// //                    tid4:source[4] + source[7] -> source[3]
// //           step 1:  tid0: source[0] + source[2] -> source[0]
// //                    tid1: source[1] + source[3] -> source[1]
// //
// //           step 2:  tid0: source[0] + source[1] -> source[0]
// // thread id: blockDim.x * blockIdx.x + threadIdx.x + step * blockDim.x * GridDim.x
// // thread 0: source[0, 8, 16, 24] sum -> shared memory



// __device__ __host__ void insert_value(float *array, int k, float data)
// {
    
//     //数值比最小的还小
//     if (data < array[k - 1])
//     {
//         return;
//     }

//     // 19, 18, 17, 16,.........4, 3, 2, 1, 0
//     for (int i = k - 2; i >= 0; i--)
//     {
//         if (data > array[i])
//         {
//             array[i + 1] = array[i];
//         }
//         else if(data == array[i]){
//             for (int j = k - 2; j > i; j--)
//             {
//                 array[j] = array[j - 1];
//             }
//             array[i] = data;
//             return;
//         }
//         else
//         {
//             array[i + 1] = data;
//             return;
//         }
//     }
//     array[0] = data;
// }


// __device__ __host__ void insert_value_cpu(float* array, int* indexes, int k, float data, int position)
// {
    
//     //数值比最小的还小
//     printf("当前运行********%d\n", position);
//     if (data < array[k - 1])
//     {
//         return;
//     }

//     // 19, 18, 17, 16,.........4, 3, 2, 1, 0
//     for (int i = k - 2; i >= 0; i--)
//     {
//         if (data > array[i])
//         {
//             array[i + 1] = array[i];
//             indexes[i + 1] = indexes[i];
//         }
//         else if(data == array[i]){
//             for (int j = k - 2; j > i; j--)
//             {
//                 array[j] = array[j - 1];
//                 indexes[j] = indexes[j - 1];
//             }
//             array[i] = data;
//             indexes[i] = position;
//             return;
//         }
//         else
//         {
//             array[i + 1] = data;
//             indexes[i+1] = position;
//             return;
//         }
//     }

//     array[0] = data;
//     indexes[0] = position;
// }


// __device__ void insert_value_gpu(float* array, int* indexes, int k, float data, int position)
// {
    
//     //数值比最小的还小
//     if (data < array[k - 1])
//     {
//         return;
//     }

//     // 19, 18, 17, 16,.........4, 3, 2, 1, 0
//     for (int i = k - 2; i >= 0; i--)
//     {
//         if (data > array[i])
//         {
//             array[i + 1] = array[i];
//             indexes[i + 1] = indexes[i];
//         }
//         else if(data == array[i]){
//             for (int j = k - 2; j > i; j--)
//             {
//                 array[j] = array[j - 1];
//                 indexes[j] = indexes[j - 1];
//             }
//             array[i] = data;
//             indexes[i] = position;
//             return;
//         }
//         else
//         {
//             array[i + 1] = data;
//             indexes[i+1] = position;
//             return;
//         }
//     }

//     array[0] = data;
//     indexes[0] = position;
// }
// __global__ void gpu_topk(float *input, float *output, int * output_indexes, int length, int k)
// {   

//     int block_size = blockDim.x;

//     // printf("**************************,%d\n ", block_size);
//     __shared__ float ken[BLOCK_SIZE * topk];
//     __shared__ float ken_idxes[BLOCK_SIZE * topk];
//     float top_array[topk];
//     int top_array_indexes[topk];

//     for (int i = 0; i < topk; i++)
//     {
//         top_array[i] = INT_MIN;
//     }

//     for (int i = 0; i < topk; i++)
//     {
//         top_array_indexes[i] = -1;
//     }

//     for (int idx = threadIdx.x + blockDim.x * blockIdx.x; idx < length; idx += gridDim.x * blockDim.x)

//     {   
//         float* pbox = input + 1 + idx *  791;
//         int keepflag = pbox[6];
//         if (keepflag){
//             // insert_value(top_array, topk, pbox[4]);
//             insert_value_gpu(top_array, top_array_indexes, topk, pbox[4], idx);
//         }
//     }


//     for (int i = 0; i < topk; i++)
//     {
//         ken[topk * threadIdx.x + i] = top_array[i];
//         ken_idxes[topk * threadIdx.x + i] = top_array_indexes[i];
//     }
//     __syncthreads();

//     for (int i = BLOCK_SIZE / 2; i >= 1; i /= 2)
//     {
//         if (threadIdx.x < i)
//         {
//             for (int m = 0; m < topk; m++)
//             {
//                 // insert_value(top_array, topk, ken[topk * (threadIdx.x + i) + m]);
//                 insert_value_gpu(top_array, top_array_indexes, topk, ken[topk * (threadIdx.x + i) + m], ken_idxes[topk * (threadIdx.x + i) + m]);
//             }
//         }
//         __syncthreads();
//         if (threadIdx.x < i)
//         {
//             for (int m = 0; m < topk; m++)
//             {
//                 ken[topk * threadIdx.x + m] = top_array[m];
//                 ken_idxes[topk * threadIdx.x + m] = top_array_indexes[m];
//             }
//         }
//         __syncthreads();
//     }
//     if (blockIdx.x * blockDim.x < length)
//     {
//         if (threadIdx.x == 0)
//         {
//             for (int i = 0; i < topk; i++)
//             {
//                 output[topk * blockIdx.x + i] = ken[i];
//                 output_indexes[topk * blockIdx.x + i] = ken_idxes[i];
//             }
//         }
//     }
// }

// __global__ void gpu_topk2(float *input, int* input_idxes, float *output, int32_t* output_indexes, int length, int k)
// {
//     __shared__ float ken[BLOCK_SIZE * topk];
//     __shared__ float ken_idxes[BLOCK_SIZE * topk];
//     float top_array[topk];
//     int top_array_indexes[topk];

//     for (int i = 0; i < topk; i++)
//     {
//         top_array[i] = INT_MIN;
//     }

//     for (int i = 0; i < topk; i++)
//     {
//         top_array_indexes[i] = -1;
//     }

//     for (int idx = threadIdx.x + blockDim.x * blockIdx.x; idx < length; idx += gridDim.x * blockDim.x)

//     {   
//         insert_value_gpu(top_array, top_array_indexes, topk, input[idx], input_idxes[idx]);
//     }


//     for (int i = 0; i < topk; i++)
//     {
//         ken[topk * threadIdx.x + i] = top_array[i];
//         ken_idxes[topk * threadIdx.x + i] = top_array_indexes[i];
//     }
//     __syncthreads();

//     for (int i = BLOCK_SIZE / 2; i >= 1; i /= 2)
//     {
//         if (threadIdx.x < i)
//         {
//             for (int m = 0; m < topk; m++)
//             {
//                 // insert_value(top_array, topk, ken[topk * (threadIdx.x + i) + m]);
//                 insert_value_gpu(top_array, top_array_indexes, topk, ken[topk * (threadIdx.x + i) + m], ken_idxes[topk * (threadIdx.x + i) + m]);
//             }
//         }
//         __syncthreads();
//         if (threadIdx.x < i)
//         {
//             for (int m = 0; m < topk; m++)
//             {
//                 ken[topk * threadIdx.x + m] = top_array[m];
//                 ken_idxes[topk * threadIdx.x + m] = top_array_indexes[m];
//             }
//         }
//         __syncthreads();
//     }
//     if (blockIdx.x * blockDim.x < length)
//     {
//         if (threadIdx.x == 0)
//         {
//             for (int i = 0; i < topk; i++)
//             {
//                 output[topk * blockIdx.x + i] = ken[i];
//                 output_indexes[topk * blockIdx.x + i] = ken_idxes[i];
//             }
//         }
//     }
// }

// void top_indexes(float* input, int32_t * output_indexes, int NMS_boxs_count, int keep_flag_boxes){

//     // float gpu_result[topk];
//     // float _1_pass_result[topk * GRID_SIZE];
//     // int _1_pass_result_idxes[topk * GRID_SIZE];

//     float* gpu_result = nullptr;
//     float* _1_pass_result = nullptr;
//     int* _1_pass_result_idxes = nullptr;

//     // 在 Device 上分配内存
//     cudaMalloc((void**)&gpu_result, keep_flag_boxes * sizeof(float));
//     cudaMalloc((void**)&_1_pass_result, keep_flag_boxes * GRID_SIZE * sizeof(float));
//     cudaMalloc((void**)&_1_pass_result_idxes, keep_flag_boxes * GRID_SIZE * sizeof(int));
    
//     gpu_topk<<<GRID_SIZE, BLOCK_SIZE>>>(input, _1_pass_result, _1_pass_result_idxes, N, keep_flag_boxes);
//     gpu_topk2<<<1, BLOCK_SIZE>>>(_1_pass_result, _1_pass_result_idxes, gpu_result, output_indexes, keep_flag_boxes * GRID_SIZE, keep_flag_boxes);
//     cudaDeviceSynchronize();
//     cudaFree(gpu_result);
//     cudaFree(_1_pass_result);
//     cudaFree(_1_pass_result_idxes);
    
// }

// __global__ void gpu_topk4(float* input, float* output, int* output_idx, int length, int k) {
//     // 在共享内存中进行计算和排序
//     __shared__ float s_data[BLOCK_SIZE];

//     int tid = threadIdx.x;
//     int block_id = blockIdx.x;
//     int idx = tid + block_id * blockDim.x;

//     if (idx < length) {
//         // 将数据从全局内存拷贝到共享内存中
//         s_data[tid] = input[idx];
//     } else {
//         s_data[tid] = -1e38; // 如果数据不存在，则设置为一个极小值
//     }

//     __syncthreads();

//     // 进行并行排序
//     for (int i = 1; i < blockDim.x; i *= 2) {
//         int j = tid - i;
        
//         // printf("循环外: i=%d, j=%d \n", i, j);
//         if (j >= 0 && s_data[j] < s_data[tid]) {
//             // printf("循环内i=%d, j=%d \n", i, j);
//             float tmp = s_data[j];
//             s_data[j] = s_data[tid];
//             s_data[tid] = tmp;
//         }
//         __syncthreads();
//     }

//     // 将排好序的数据和索引值写回全局内存
//     if (tid < k) {
//         output[block_id * k + tid] = s_data[tid];
//         for (int i = 0; i < length; i++) {
//             if (input[i] == s_data[tid]) {
//                 output_idx[block_id * k + tid] = i;
//                 break;
//             }
//         }
//     }
// }


// __global__ void gpu_topk5(float* input, float* output, int* output_idx, int length, int k) {
//     // 在共享内存中进行计算和排序
//     __shared__ float s_data[50];
//     __shared__ int s_data_idx[50];

//     int position = threadIdx.x + blockIdx.x * blockDim.x;
//     if (position >= length) return;
//     float data = input[position];

//     // 在共享内存中进行插入排序
//     for (int i = 0; i < k; i++) {
//         if (data > s_data[i]) {
//             for (int j = k - 1; j > i; j--) {
//                 s_data[j] = s_data[j - 1];
//                 s_data_idx[j] = s_data_idx[j - 1];
//             }
//             s_data[i] = data;
//             s_data_idx[i] = position;
//             break;
//         }
//     }

//     __syncthreads();

//     if (threadIdx.x == 0) {
//         // 将结果保存到全局内存
//         for (int i = 0; i < k; i++) {
//             output[i] = s_data[i];
//             output_idx[i] = s_data_idx[i];
//         }
//     }
// }



// void cpu_topk(float * &input, float* output, int* &indexes, int length, int k)
// {
//     for (int i = 0; i < length; i++)
//     {   
//         float* pbox = input + 1 + i *  791;
//         int keepflag = pbox[6];
//         if (keepflag){
//             insert_value_cpu(output, indexes, k, pbox[4], i);
//         }
//     }
// }

// int main()
// {   
//     srand(1); // 设置种子
//     printf("Init source data...........\n");
//     for (int i = 0; i < N; i++)
//     {
//         source[i] = float(rand()) / RAND_MAX;
//         printf("当前值:%f\n", source[i]);
//     }

//     // cudaStream_t stream_ = nullptr;
//     TRT::Tensor output(TRT::DataType::Float);
//     // output.set_stream(stream_);

//     output.load_from_file("/media/ps/data/train/LQ/LQ/bdmask4/workspace/mask_data/tdata/out_array_20");
    
//     float * output_array_ptr_cpu = output.cpu<float>();
//     // output.to_gpu(); 
//     float * output_array_ptr = output.gpu<float>();


//     printf("Complete init source data.....\n");
//     cudaEvent_t start, stop_gpu, stop_cpu, start_cpu;
//     cudaEventCreate(&start);
//     cudaEventCreate(&stop_gpu);
//     cudaEventCreate(&stop_cpu);
//     cudaEventCreate(&start_cpu);
//     TRT::Tensor keepflag_indexes(TRT::DataType::Int32);
//     keepflag_indexes.resize(1, topk).to_gpu();
//     int32_t* keepflag_indexes_ptr = keepflag_indexes.gpu<int32_t>();
//     cudaEventRecord(start);
//     cudaEventSynchronize(start);
//     printf("GPU Run **************\n");
//     int cr = 1;    
//     cudaMemsetAsync(gpu_result, 0, sizeof(float)*topk);
//     cudaMemsetAsync(gpu_result_idxes, 0, sizeof(int)*topk);
//     for (int i = 0; i < cr; i++)

//     {   

//         // gpu_topk<<<GRID_SIZE, BLOCK_SIZE>>>(output_array_ptr, _1_pass_result, _1_pass_result_idxes, N, topk);

//         // gpu_topk2<<<1, BLOCK_SIZE>>>(_1_pass_result, _1_pass_result_idxes, gpu_result, gpu_result_idxes, topk * GRID_SIZE, topk);


//         // cudaDeviceSynchronize();

//         // top_indexes(output_array_ptr, keepflag_indexes_ptr, N, topk);
//         // gpu_topk4<<<4, BLOCK_SIZE>>>(source, gpu_result, gpu_result_idxes, N, topk);
//         gpu_topk5<<<4, BLOCK_SIZE>>>(source, gpu_result, gpu_result_idxes, N, topk);

//     }
//     float threshold = 0.15;
//     printf("GPU Complete!!!\n");
//     cudaEventRecord(stop_gpu);
//     cudaEventSynchronize(stop_gpu);
//     keepflag_indexes.to_cpu();
//     int32_t* keepflag_indexes_cpu = keepflag_indexes.cpu<int32_t>();


//     float cpu_result[topk] = {0};
//     TRT::Tensor box_output_device(TRT::DataType::Float);
//     float* box_output_cpu = box_output_device.cpu<float>();
//     box_output_device.resize(topk, 2 + 6).to_gpu();  // counter, left, right, top, bottom, conf, label
//     TRT::Tensor indexes(TRT::DataType::Int32);
//     indexes.resize(1, topk);
//     int32_t* indexes_cpu = indexes.cpu<int32_t>();
//     memset(indexes_cpu, -1, topk*sizeof(int32_t));
//     printf("CPU RUN***************\n");
    
//     auto st = iLogger::timestamp_now_float();
//     cudaEventRecord(start_cpu);
//     for (int t=0; t < cr; t++){
//         memset(cpu_result, 0, topk*sizeof(float));
//         memset(indexes_cpu, -1, topk*sizeof(int32_t));
//         cpu_topk(output_array_ptr_cpu, cpu_result, indexes_cpu,  N, topk);
//         //  int count = count_if(indexes_cpu, indexes_cpu+topk, [](int x){return x >= 0;});


//         // printf("***********************************************************************\n");
//     }
//     auto et = iLogger::timestamp_now_float();

//     int count = count_if(indexes_cpu, indexes_cpu+topk, [](int x){return x >= 0;});
//     // for (int i = 0; i < topk; i++)
//     // {
//     //     printf("共有%d个缺陷，CPU top%d: %f; index top%d: %d\n", count, i + 1, cpu_result[i], i + 1, indexes_cpu[i]);
//     // }

//     // cpu_sort2(output_array_ptr_cpu, N, topk, cpu_result);
//     cudaEventRecord(stop_cpu);
//     cudaEventSynchronize(stop_cpu);
//     printf("CPU Complete!!!!!\n");

//     float time_cpu, time_gpu;
//     cudaEventElapsedTime(&time_gpu, start, stop_gpu);
//     cudaEventElapsedTime(&time_cpu, start_cpu, stop_cpu);

//     bool error = false;
//     for (int i = 0; i < topk; i++)
//     {
//         printf("GPU top%d: %f---%d\n",  i + 1, gpu_result[i], gpu_result_idxes[i]);
//         // printf("GPU top%d: %d\n",  i + 1,  keepflag_indexes_cpu[i]);
        
//         // printf("GPU top%d: %f---%d\n",  i + 1, _1_pass_result[i], _1_pass_result_idxes[i]);
//         // if (fabs(gpu_result[i] - cpu_result[i]) > 0)
//         // {
//         //     error = true;
//         // }
//     }
//     printf("Result: %s\n", (error ? "Error" : "Pass"));
//     printf("CPU time: %.5f; GPU time: %.5f\n", (et-st)/cr, (time_gpu / cr));
//     printf("CPU time: %.5f; GPU time: %.5f\n", (time_cpu/cr), (time_gpu / cr));
// }
