#include <common/cuda_tools.hpp>

namespace EfficientAd{

    static __global__ void decode_kernel(float* predict, int dst_height, int dst_width, float confidence_threshold, uint8_t* parray, int jobs){  

        int position = blockDim.x * blockIdx.x + threadIdx.x;
        if (position >= jobs) return;
        float* pitem = predict + position;
        uint8_t* pout_item = parray + position;
        float pout_item_value =  (*pitem < 0.0f) ? 0.0f : ((*pitem > 1.0f) ? 1.0f : *pitem);  
        float pout_value = floorf(pout_item_value * 255.0f + 0.5f);
        *pout_item = (pout_value > confidence_threshold) ? 255 : 0;
    }

    void decode_kernel_invoker(float* predict, int dst_height, int dst_width, float confidence_threshold, uint8_t* parray, cudaStream_t stream){
        auto jobs = dst_height * dst_width;
        auto grid = CUDATools::grid_dims(jobs);
        auto block = CUDATools::block_dims(jobs);
        checkCudaKernel(decode_kernel<<<grid, block, 0, stream>>>(predict, dst_height, dst_width, confidence_threshold, parray, jobs));
    }
};