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


// static __device__ void affine_project(float* matrix, float x, float y, float* ox, float* oy){
//     *ox = matrix[0] * x + matrix[1] * y + matrix[2];
//     *oy = matrix[3] * x + matrix[4] * y + matrix[5];
// }

// static __global__ void decode_kernel(float* predict, int num_bboxes, int num_classes, float confidence_threshold, 
//                                     float* invert_affine_matrix, float* parray, int max_objects){  

//     int position = blockDim.x * blockIdx.x + threadIdx.x;
//     if (position >= num_bboxes) return;

//     float* pitem     = predict + (5 + num_classes + 784) * position;
//     float objectness = pitem[4];
//     if(objectness < confidence_threshold)
//         return;

//     float* class_confidence = pitem + 5;
//     float confidence        = *class_confidence++;
//     int label               = 0;
//     for(int i = 1; i < num_classes; ++i, ++class_confidence){
//         if(*class_confidence > confidence){
//             confidence = *class_confidence;
//             label      = i;
//         }
//     }

//     confidence = objectness;
//     if(confidence < confidence_threshold)
//         return;

//     int index = atomicAdd(parray, 1);
//     if(index >= max_objects)
//         return;

//     float left       = *pitem++;
//     float top        = *pitem++;
//     float right      = *pitem++;
//     float bottom     = *pitem++;
//     affine_project(invert_affine_matrix, left,  top,    &left,  &top);
//     affine_project(invert_affine_matrix, right, bottom, &right, &bottom);

//     float* pout_item = parray + 1 + index * 791;
//     *pout_item++ = left;
//     *pout_item++ = top;
//     *pout_item++ = right;
//     *pout_item++ = bottom;
//     *pout_item++ = confidence;
//     *pout_item++ = label;
//     *pout_item++ = 1; // 1 = keep, 0 = ignore


//     for(int i = 0; i < 784; ++i){
//         *pout_item++ = *(pitem+i+1+num_classes);
//     }
// }


// void decode_kernel_invoker(float* predict, int num_bboxes, int num_classes, float confidence_threshold, float* invert_affine_matrix, float* parray, int max_objects, cudaStream_t stream){
    
//     auto grid = CUDATools::grid_dims(num_bboxes);
//     auto block = CUDATools::block_dims(num_bboxes);
//     checkCudaKernel(decode_kernel<<<grid, block, 0, stream>>>(predict, num_bboxes, num_classes, confidence_threshold, invert_affine_matrix, parray, max_objects));
// }


// int main()
// {
//     cudaStream_t stream_ = nullptr;
//     TRT::Tensor output(TRT::DataType::Float);
//     output.set_stream(stream_);

//     output.load_from_file("/media/ps/data/train/LQ/LQ/bdmask4/workspace/mask_data/tdata/2-orig-output_2");
    
//     float * output_ptr_cpu = output.cpu<float>();
//     output.to_gpu(); 
//     float * output_ptr = output.gpu<float>();
//     printf("*******************");

//     // TRT::Tensor affin_matrix_device(TRT::DataType::Float);
//     // affin_matrix_device.set_stream(stream_);
//     // affin_matrix_device.load_from_file("/media/ps/data/train/LQ/LQ/bdmask4/workspace/mask_data/tdata/2-affin_matrix_device_2");
//     // affin_matrix_device.to_gpu(false);
//     // float * affin_matrix_ptr = affin_matrix_device.gpu<float>();

//     // TRT::Tensor output_array_device(TRT::DataType::Float);
//     // output_array_device.set_stream(stream_);
//     // output_array_device.resize(1, 1 + 1024 * 791).to_gpu(); 
//     // output_array_device.to_gpu(false);
//     // float * output_array_ptr = output_array_device.gpu<float>();
//     // checkCudaRuntime(cudaMemsetAsync(output_array_ptr, 0, sizeof(float), stream_));

//     // int num_bboxes = 87296;
//     // auto grid = CUDATools::grid_dims(num_bboxes);
//     // auto block = CUDATools::block_dims(num_bboxes);
//     // decode_kernel<<<grid, block, 0, stream_>>>(output_ptr, num_bboxes, 25, 0.15, affin_matrix_ptr, output_array_ptr, 1024);
//     // decode_kernel_invoker(output_ptr, 87296, 25, 0.15, affin_matrix_ptr, output_array_ptr, 1024, stream_);
//     // float * output_array_cpu = output_array_device.cpu<float>();

//     // int num = 1;
//     // // for(auto &idx: idxes){
//     // for(int idx = 0; idx < 87296; idx++){
//     //     auto *pitem = output_ptr_cpu + idx * 814;
//     //     float score = pitem[4];
//     //     // printf("%f\n", score);
//     //     if (score > 0.15){
//     //         int x1=pitem[0], y1=pitem[1], x2=pitem[2], y2 =pitem[3];
//     //         printf("top%d--当前索引%d--box信息: %d-%f-%d-%d-%d\n", num, idx, sqrt(score), x1, y1, x2, y2);
//     //         num++;
//     //     } 
//     // }

//     // vector<int> idxes = {10, 35, 52, 26, 8, 15, 20, 47, 34, 59, 27, 57, 23, 33, 12, 3, 25, 6, 17};
//     // int num = 1;
//     // // for(auto &idx: idxes){
//     // for(int idx = 0; idx < 1024; idx++){
//     //     auto *pitem = output_array_cpu + 1 + idx *  791;
//     //     int keepflag = pitem[6];
//     //     if (keepflag){
//     //         float score = pitem[4];
//     //         int x1=pitem[0], y1=pitem[1], x2=pitem[2], y2 =pitem[3];
//     //         int label = pitem[5];
//     //         printf("%d---top%d--当前索引%d--box信息: %d-%f-%d-%d-%d-%d\n", int(*output_array_cpu), num, idx, label, sqrt(score), x1, y1, x2, y2);
//     //         num++;
//     //     } 
//     // }
//     return 0;
// }
