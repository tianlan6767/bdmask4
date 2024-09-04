#include <common/cuda_tools.hpp>
#include "cfaAd.hpp"


namespace CfaAd{

    static __device__ bool isInsideDisk(int x, int y, int r) {
        return (x * x + y * y) <= (r * r);
    }


    static __global__ void decode_kernel(float* predict, int dst_height, int dst_width, float confidence_threshold, uint8_t* parray, int jobs){  

        int position = blockDim.x * blockIdx.x + threadIdx.x;
        if (position >= jobs) return;
        float* pitem = predict + position;
        uint8_t* pout_item = parray + position;
        float pout_item_value =  (*pitem < 0.0f) ? 0.0f : ((*pitem > 1.0f) ? 1.0f : *pitem);  
        float pout_value = floorf(pout_item_value * 255.0f + 0.5f);
        *pout_item = (pout_value > confidence_threshold) ? pout_value : 0;
        // *pout_item = (pout_value > confidence_threshold) ? 255 : 0;
    }

    static __global__ void gaussian_filter_kernel(const float* input, float* output, int width, int height, const float* kernel, int jobs){  
        int position = blockIdx.x * blockDim.x + threadIdx.x;

        int x = position % width;
        int y = position / width;

        if (x >= width || y >= height) {
            return;
        }

        float sum = 0.0f;
        float weight_sum = 0.0f;

        // 遍历高斯核
        for (int i = -KERNEL_SIZE / 2; i <= KERNEL_SIZE / 2; ++i) {
            for (int j = -KERNEL_SIZE / 2; j <= KERNEL_SIZE / 2; ++j) {
            int dx = x + i;
            int dy = y + j;

            // 检查是否在图像边界内
            if (dx >= 0 && dx < width && dy >= 0 && dy < height) {
                int k_idx = (i + KERNEL_SIZE / 2) * KERNEL_SIZE + (j + KERNEL_SIZE / 2);
                sum += static_cast<float>(input[dy * width + dx]) * kernel[k_idx];
                weight_sum += kernel[k_idx];
            }
            }
        }

        output[y * width + x] = sum / weight_sum;
    }

    // Kernel function to find the maximum and minimum values in an array
    static __global__ void findMinMaxKernel(const float* data,  float* maxResult, float* minResult,  size_t n) {
        extern __shared__ float sharedData[];

        unsigned int tid = threadIdx.x;
        unsigned int index = blockIdx.x * blockDim.x + tid;

        // Load elements into shared memory
        if (index < n) {
            sharedData[tid] = data[index];                   // For max computation
            sharedData[blockDim.x + tid] = data[index];      // For min computation
        } else {
            sharedData[tid] = -FLT_MAX;
            sharedData[blockDim.x + tid] = FLT_MAX;
        }
        __syncthreads();

        // Reduction to find the maximum and minimum
        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s && index + s < n) {
                sharedData[tid] = fmaxf(sharedData[tid], sharedData[tid + s]);
                sharedData[blockDim.x + tid] = fminf(sharedData[blockDim.x + tid], sharedData[blockDim.x + tid + s]);
            }
            __syncthreads();
        }

        // Write the result for this block to global memory
        if (tid == 0) {
            maxResult[blockIdx.x] = sharedData[0];
            minResult[blockIdx.x] = sharedData[blockDim.x];
        }
    }
        
    // Define a kernel function to normalize the data
    static __global__ void simpleNormalizeKernel(float* data, uint8_t* output, float* max_result, float * min_result, int result_size,  float confidence_threshold, int jobs) {
        // Get the thread ID


        float min_val = min_result[0];
        float max_val = max_result[0];
        for (int i = 0 ; i < result_size; ++i){
            if (min_val > min_result[i])
                min_val = min_result[i];
            if(max_val < max_result[i]){
                max_val = max_result[i];
            }
        }

        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        // Check if the thread ID is within the array bounds
        if (idx < jobs) {
            // Normalize the data using the min and max values
            data[idx] = (data[idx] - min_val) / (max_val - min_val);
            // output[idx] = (data[idx] > confidence_threshold) ? 255 : 0;
            output[idx] = (data[idx] > confidence_threshold) ? data[idx]*255 : 0 ;
        }
    }
    

    static __global__ void erodeInPlaceKernel(uint8_t* mask, int width, int height, int r, int jobs) {

            int position = blockIdx.x * blockDim.x + threadIdx.x;
            if (position > jobs) return;
            
            int x = position % width;
            int y = position / width;
            int half_r = r;
            bool isForeground = true;

            for (int dy = -half_r; dy <= half_r; ++dy) {
                for (int dx = -half_r; dx <= half_r; ++dx) {
                    int nx = x + dx;
                    int ny = y + dy;

                    if (nx >= 0 && nx < width && ny >= 0 && ny < height && isInsideDisk(dx, dy, r)) {
                        if (mask[ny * width + nx] == 0) {
                            isForeground = false;
                            break;
                        }
                    }
                }
                if (!isForeground) break;
            }

            if (!isForeground) {
                mask[y * width + x] = 0;
            }
    }


    static __global__ void dilateKernel(uint8_t* input, uint8_t* output, int width, int height, int r, int jobs) {
        int position = blockIdx.x * blockDim.x + threadIdx.x;
        if (position > jobs) return;
        
        int x = position % width;
        int y = position / width;

        int half_r = r;
        bool isForeground = false;
        int tmp_max = 0;
        for (int dy = -half_r; dy <= half_r; ++dy) {
            for (int dx = -half_r; dx <= half_r; ++dx) {
                int nx = x + dx;
                int ny = y + dy;

                if (nx >= 0 && nx < width && ny >= 0 && ny < height && isInsideDisk(dx, dy, r)) {
                    if (input[ny * width + nx] > tmp_max) {
                        isForeground = true;
                        tmp_max = input[ny * width + nx];
                        // break;
                    }
                }
            }
            // if (isForeground) break;
        }
        output[y * width + x] = isForeground ? tmp_max : 0;
        
        // output[y * width + x] = isForeground ? 255 : 0;
    }


    void findMinMax(float * data, float *max_result, float * min_result, int size, cudaStream_t stream){
        // Allocate device memory

        auto jobs = size;
        auto grid = CUDATools::grid_dims(jobs);
        auto block = CUDATools::block_dims(jobs);
        checkCudaKernel(findMinMaxKernel<<<grid, block, block.x *sizeof(float)*2, stream>>>(data, max_result, min_result, jobs));
    }


    void simple_normalize(float*data, uint8_t *output, float*max_result, float * min_result, int size, int min_max_result_size, float confidence_threshold,  cudaStream_t stream){
        auto jobs = size;
        auto grid = CUDATools::grid_dims(jobs);
        auto block = CUDATools::block_dims(jobs);
        checkCudaKernel(simpleNormalizeKernel<<<grid, block, 0, stream>>>(data, output, max_result, min_result, grid.x, confidence_threshold, jobs));
    }


    void gaussian_filter(float * input, float * output, int width, int height, const float* kernel, cudaStream_t stream){
        auto jobs = width * height;
        auto grid = CUDATools::grid_dims(jobs);
        auto block = CUDATools::block_dims(jobs);
        checkCudaKernel(gaussian_filter_kernel<<<grid, block, 0, stream>>>(input, output, width, height, kernel, jobs));
    }

    void decode_kernel_invoker(float* predict, int dst_height, int dst_width, float confidence_threshold, uint8_t* parray, cudaStream_t stream){
        auto jobs = dst_height * dst_width;
        auto grid = CUDATools::grid_dims(jobs);
        auto block = CUDATools::block_dims(jobs);
        checkCudaKernel(decode_kernel<<<grid, block, 0, stream>>>(predict, dst_height, dst_width, confidence_threshold, parray, jobs));
    }

    void morphologyOpening(uint8_t * mask_array_ptr, uint8_t* mask_out_ptr, int width, int height, int radius, cudaStream_t stream){
        auto jobs = width * height;
        auto grid = CUDATools::grid_dims(jobs);
        auto block = CUDATools::block_dims(jobs);
        
        checkCudaKernel(erodeInPlaceKernel<<<grid, block, 0, stream>>>(mask_array_ptr, height, width, 3, jobs));
        checkCudaKernel(dilateKernel<<<grid, block, 0, stream>>>(mask_array_ptr, mask_out_ptr, width, height, 5, jobs));
    }
};