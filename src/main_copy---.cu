#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <common/cuda_tools.hpp>
#include <common/trt_tensor.hpp>

__device__ float bilinear_interpolate(const float* bottom_data,const int height,const int width,float y,float x, const int index /* index for debug only*/) {
    // deal with cases that inverse elements are out of feature map boundary
    if (y < -1.0 || y > height || x < -1.0 || x > width) {
        // empty
        return 0;
    }

    if (y <= 0)
        y = 0;
    if (x <= 0)
        x = 0;

    int y_low = (int)y;
    int x_low = (int)x;
    int y_high;
    int x_high;

    if (y_low >= height - 1) {
        y_high = y_low = height - 1;
        y = (float)y_low;
    } else {
        y_high = y_low + 1;
    }

    if (x_low >= width - 1) {
        x_high = x_low = width - 1;
        x = (float)x_low;
    } else {
        x_high = x_low + 1;
    }

    float ly = y - y_low;
    float lx = x - x_low;
    float hy = 1. - ly, hx = 1. - lx;
    // do bilinear interpolation
    float v1 = bottom_data[y_low * width + x_low];
    float v2 = bottom_data[y_low * width + x_high];
    float v3 = bottom_data[y_high * width + x_low];
    float v4 = bottom_data[y_high * width + x_high];
    float w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

    float val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

    return val;
}


__global__ void decode_interpolate_kernel(float* input, const int channels, const int input_height, const int input_width,
                                   float* output, const int output_height, const int output_width,
                                   const float scale_factor, int jobs) {
    int index = (blockIdx.x * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
    if (index > jobs) return;
    int output_y = (index / output_width) % output_height;
    int output_x = index % output_width;
    int c = (index / output_width / output_height) % channels;
    int n = index / output_width / output_height / channels; 

    float y = (float)((output_y + 0.5f) * scale_factor - 0.5f);
    float x = (float)((output_x + 0.5f) * scale_factor - 0.5f);
    const float* offset_bottom_data =
        input + (n * channels + c) * input_height * input_width;
    output[index] = bilinear_interpolate(offset_bottom_data, input_height, input_width, y, x, index);
}


__global__ void softmax_kernel(float* input, const int height, const int width, int jobs) {
    int index = (blockIdx.x * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
    if (index > jobs) return;
    int dy = index / width;
    int dx = index % width;
    int area = height * width;
    float* p_c0 = input + dy * width + dx;
    float* p_c1 = p_c0 + area;
    float* p_c2 = p_c1 + area;
    float* p_c3 = p_c2 + area;
    float c0 = *p_c0;
    float c1 = *p_c1;
    float c2 = *p_c2;
    float c3 = *p_c3;
    float max_val = fmaxf(fmaxf(c0, c1), fmaxf(c2, c3));
    auto sum_exp = expf(c0) + expf(c1) + expf(c2)+ expf(c3);

    *p_c0 = expf(c0 - max_val) / sum_exp;
    *p_c1 = expf(c1 - max_val) / sum_exp;
    *p_c2 = expf(c2 - max_val) / sum_exp;
    *p_c3 = expf(c3 - max_val) / sum_exp;
}

__global__ void mul_sum_sigmod_kernel(float* top_data, float*feat_out_tensor, int height, int width, float* mask_pred, int jobs) {
    
    int index = (blockIdx.x * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
    if (index > jobs) return;
    int dy = index / width;
    int dx = index % width;
    int area = height * width;
    float* p_c0 = top_data + dy * width + dx;
    float* p_c1 = p_c0 + area;
    float* p_c2 = p_c1 + area;
    float* p_c3 = p_c2 + area;
    float* f_c0 = feat_out_tensor + dy * width + dx;
    float* f_c1 = f_c0 + area;
    float* f_c2 = f_c1 + area;
    float* f_c3 = f_c2 + area;
    float* dst_mask = mask_pred + dy * width + dx;
    *dst_mask = (*p_c0) * (*f_c0) + (*p_c1) * (*f_c1) + (*p_c2) * (*f_c2)+ (*p_c3) * (*f_c3);
}


__global__ void RoIAlignForward(
    const float* bottom_data,
    const float spatial_scale,
    const int channels,
    const int height,
    const int width,
    const int pooled_height,
    const int pooled_width,
    const int sampling_ratio,
    const float* bottom_rois,
    float* top_data,
    bool aligned) {
    // (n, c, ph, pw) is an element in the pooled output
    int index = (blockIdx.x * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    const float* offset_bottom_rois = bottom_rois + n * 5;
    int roi_batch_ind = offset_bottom_rois[0];

    // Do not use rounding; this implementation detail is critical
    float offset = aligned ? (float)0.5 : (float)0.0;
    float roi_start_w = offset_bottom_rois[1] * spatial_scale - offset;
    float roi_start_h = offset_bottom_rois[2] * spatial_scale - offset;
    float roi_end_w = offset_bottom_rois[3] * spatial_scale - offset;
    float roi_end_h = offset_bottom_rois[4] * spatial_scale - offset;

    float roi_width = roi_end_w - roi_start_w;
    float roi_height = roi_end_h - roi_start_h;
    if (!aligned) { // for backward-compatibility only
      roi_width = max(roi_width, (float)1.);
      roi_height = max(roi_height, (float)1.);
    }
    float bin_size_h = static_cast<float>(roi_height) / static_cast<float>(pooled_height);
    float bin_size_w = static_cast<float>(roi_width) / static_cast<float>(pooled_width);

    const float* offset_bottom_data =
        bottom_data + (roi_batch_ind * channels + c) * height * width;

    // We use roi_bin_grid to sample the grid and mimic integral
    int roi_bin_grid_h = (sampling_ratio > 0)
        ? sampling_ratio
        : ceil(roi_height / pooled_height); // e.g., = 2
    int roi_bin_grid_w =
        (sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / pooled_width);

    // We do average (integral) pooling inside a bin
    // When the grid is empty, output zeros == 0/1, instead of NaN.
    const float count = max(roi_bin_grid_h * roi_bin_grid_w, 1); // e.g. = 4

    float output_val = 0.;
    for (int iy = 0; iy < roi_bin_grid_h; iy++) // e.g., iy = 0, 1
    {
      const float y = roi_start_h + ph * bin_size_h +
          static_cast<float>(iy + .5f) * bin_size_h /
              static_cast<float>(roi_bin_grid_h); // e.g., 0.5, 1.5
      for (int ix = 0; ix < roi_bin_grid_w; ix++) {
        const float x = roi_start_w + pw * bin_size_w +
            static_cast<float>(ix + .5f) * bin_size_w /
                static_cast<float>(roi_bin_grid_w);
        // printf("%f, %f, %f, %f, %f, %f, %f, %f, %f, %f\n", roi_height, roi_width, roi_start_h, roi_start_w, roi_bin_grid_h, roi_bin_grid_w, bin_size_h, bin_size_w, x, y);
        float val = bilinear_interpolate(
            offset_bottom_data, height, width, y, x, index);
        output_val += val;
      }
    }

    output_val /= count;
    top_data[index] = output_val;
}

__global__ void generate_grid_kernel(const float* box, int N, int h, int w, float* box_grid, int jobs) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= jobs) return;

    int n = idx / (h * w);
        
    int x = (idx % (h * w)) % w;
    int y = (idx % (h * w)) / w;

    float img_x = x + int(box[n * 5 + 1]) + 0.5f;
    float img_y = y + int(box[n * 5 + 2]) + 0.5f;

    img_x = (img_x - box[n * 5 + 1]) / (box[n * 5 + 3] - box[n * 5 + 1]) * 2.0f - 1.0f;
    img_y = (img_y - box[n * 5 + 2]) / (box[n * 5 + 4] - box[n * 5 + 2]) * 2.0f - 1.0f;

    box_grid[idx * 2] = img_x;
    box_grid[idx * 2 + 1] = img_y;
}

__global__ void decode_grid_sample_kernel(float* mask, int mask_height, int mask_width, float* box_grid,  int grid_h, int grid_w, uint8_t* box_mask, int jobs){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= jobs) return;
    float grid_x = (mask_width / 2.0f) * ((box_grid[idx*2]) + 1) ;
    float grid_y = (mask_height / 2.0f) * (box_grid[idx*2 + 1] + 1);
    // printf("%d-%f-%f-%f\n", idx, grid_y, grid_x, (bilinear_interpolate(mask, mask_height, mask_width, grid_y, grid_x, idx)));
    box_mask[idx] = (bilinear_interpolate(mask, mask_height, mask_width, grid_y, grid_x, idx)) > 0.5f ? 255 : 0;
}

static void decode_grid_sample(float* mask, int mask_height, int mask_width, float* box_grid,  int grid_height, int grid_width, uint8_t* box_mask, cudaStream_t stream){
    int jobs   = grid_height * grid_width;
    auto grid  = CUDATools::grid_dims(jobs);
    auto block = CUDATools::block_dims(jobs);
    checkCudaKernel(decode_grid_sample_kernel<<<grid, block, 0, stream>>>(
            mask, mask_height, mask_width, box_grid, grid_height, grid_width, box_mask, jobs
    ));
}

static void decode_softmax(float* input, int height, int width, cudaStream_t stream) {
    int jobs   = width * height;
    auto grid  = CUDATools::grid_dims(jobs);
    auto block = CUDATools::block_dims(jobs);
    checkCudaKernel(softmax_kernel<<<grid, block, 0, stream>>>(
        input, height, width, jobs
    ));
}

static void generate_grid(float* box, int N, int height, int width, float* box_grid, cudaStream_t stream){
    int jobs   = N * width * height;
    auto grid  = CUDATools::grid_dims(jobs);
    auto block = CUDATools::block_dims(jobs);
    checkCudaKernel(generate_grid_kernel<<<grid, block, 0, stream>>>(
            box, N, height, width, box_grid, jobs
    ));
}

static void decode_mul_sum_sigmod(float * top_data, float * feat_out_tensor, int height, int width,  float*mask_pred, cudaStream_t stream){
    int jobs   = width * height;
    auto grid  = CUDATools::grid_dims(jobs);
    auto block = CUDATools::block_dims(jobs);
    checkCudaKernel(mul_sum_sigmod_kernel<<<grid, block, 0, stream>>>(
            top_data, feat_out_tensor, height, width, mask_pred, jobs
    ));
}

static void decode_roialign(float* bottom_data,float spatial_scale, int channels, int height, int width, int pooled_height, int pooled_width, float sampling_ratio, float* bottom_rois,
      float* top_data, bool aligned, cudaStream_t stream) {
    int jobs   = channels * pooled_height * pooled_width;
    auto grid  = CUDATools::grid_dims(jobs);   
    auto block = CUDATools::block_dims(jobs);
    checkCudaKernel(RoIAlignForward<<<grid, block, 0, stream>>>(
        bottom_data, spatial_scale, channels, height, width, pooled_height, pooled_width, sampling_ratio, bottom_rois,
        top_data, aligned));
}

static void decode_interpolate(float* src, int channels, int src_height, int src_width, float* dst, int dst_height, int dst_width,float scale_factor,
        cudaStream_t stream) {
        int jobs   = channels * dst_width * dst_height;
        auto grid  = CUDATools::grid_dims(jobs);
        auto block = CUDATools::block_dims(jobs);
        checkCudaKernel(decode_interpolate_kernel<<<grid, block, 0, stream>>>(
            src, channels, src_height, src_width, dst, dst_height, dst_width, scale_factor, jobs));
    }

int main() {

    // 初始化 CUDA
    cudaStream_t stream_ = nullptr;
    cudaSetDevice(3);
    // 定义时间戳变量
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    TRT::Tensor base_device(TRT::DataType::Float);
    TRT::Tensor box_device(TRT::DataType::Float);
    TRT::Tensor feat_device(TRT::DataType::Float);
    TRT::Tensor feat_out_device(TRT::DataType::Float);
    TRT::Tensor top_device(TRT::DataType::Float);
    TRT::Tensor mask_pred_device(TRT::DataType::Float);
    TRT::Tensor box_grid_device(TRT::DataType::Float);
    TRT::Tensor grid_w_device(TRT::DataType::Float);
    TRT::Tensor grid_h_device(TRT::DataType::Float);
    TRT::Tensor box_mask_device(TRT::DataType::UInt8);
    base_device.load_from_file("/media/ps/data/train/LQ/LQ/bdms/bdmask/workspace/CK/inf/base_input");
    box_device.load_from_file("/media/ps/data/train/LQ/LQ/bdms/bdmask/workspace/CK/inf/box_input");
    feat_device.load_from_file("/media/ps/data/train/LQ/LQ/bdms/bdmask/workspace/CK/inf/top_feat_input");
    
    float* box_cpu = box_device.cpu<float>();
    int box_mask_height = box_cpu[4] - box_cpu[2] + 0.5f;
    int box_mask_width  = box_cpu[3] - box_cpu[1] + 0.5f;
        
    // 启动计时
    cudaEventRecord(start, stream_);
    
    box_grid_device.resize(1, box_mask_height, box_mask_width, 2);
    top_device.resize(1,4,56,56);
    feat_device.resize(1,4,14,14);
    feat_out_device.resize(1, 4, 56, 56);
    mask_pred_device.resize(1, 1, 56, 56);
    grid_w_device.resize(1, box_mask_width);
    grid_h_device.resize(1, box_mask_height);
    box_mask_device.resize(box_mask_height, box_mask_width);


    float* box_tensor = box_device.gpu<float>();
    float* feat_tensor = feat_device.gpu<float>();
    float* feat_out_tensor = feat_out_device.gpu<float>();
    float* base_tensor = base_device.gpu<float>();
    float* top_data = top_device.gpu<float>();
    float* mask_pred = mask_pred_device.gpu<float>();
    float* box_grid = box_grid_device.gpu<float>();
    uint8_t* box_mask = box_mask_device.gpu<uint8_t>();
    

    int channels = base_device.size(1);
    int base_height = base_device.size(2);
    int base_width = base_device.size(3);

    float spatial_scale = 0.25f;
    int pooled_height = 56;
    int pooled_width = 56;
    int sampling_ratio = 1;
    bool aligned = true;

    // 调用 CUDA 核函数   
    generate_grid(box_tensor, 1, box_mask_height, box_mask_width, box_grid, stream_);
    decode_roialign(base_tensor, spatial_scale, channels, base_height, base_width, pooled_height, pooled_width, sampling_ratio, box_tensor, top_data, aligned, stream_);
    decode_interpolate(feat_tensor, 4, 14, 14, feat_out_tensor, 56, 56, 0.25, stream_);
    decode_softmax(feat_out_tensor, 56, 56, stream_);
    decode_mul_sum_sigmod(top_data, feat_out_tensor, 56, 56,  mask_pred, stream_);
    decode_grid_sample(mask_pred, 56, 56, box_grid,  box_mask_height, box_mask_width, box_mask, stream_);
    
    // 停止计时
    cudaEventRecord(stop, stream_);
    cudaEventSynchronize(stop);

    // 计算执行时间
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    top_device.save_to_file("/media/ps/data/train/LQ/LQ/bdms/bdmask/workspace/CK/inf/top_input-roialigned");
    feat_device.save_to_file("/media/ps/data/train/LQ/LQ/bdms/bdmask/workspace/CK/inf/feat_resize");
    feat_out_device.save_to_file("/media/ps/data/train/LQ/LQ/bdms/bdmask/workspace/CK/inf/feat_out_device");
    mask_pred_device.save_to_file("/media/ps/data/train/LQ/LQ/bdms/bdmask/workspace/CK/inf/mask_pred_device");
    box_grid_device.save_to_file("/media/ps/data/train/LQ/LQ/bdms/bdmask/workspace/CK/inf/box_grid_device");
    grid_h_device.save_to_file("/media/ps/data/train/LQ/LQ/bdms/bdmask/workspace/CK/inf/grid_h_device");
    grid_w_device.save_to_file("/media/ps/data/train/LQ/LQ/bdms/bdmask/workspace/CK/inf/grid_w_device");
    box_mask_device.save_to_file("/media/ps/data/train/LQ/LQ/bdms/bdmask/workspace/CK/inf/box_mask_device");

    // 打印执行时间
    std::cout << "CUDA Kernel Execution Time: " << elapsedTime << " ms" << std::endl;

    // 释放时间戳资源
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    // float left, top, right, bottom;
    // left = box_tensor[1];
    // top = box_tensor[2];
    // right = box_tensor[3];
    // bottom = box_tensor[4];
    // int box_width = right - left + 0.5f;
    // int box_height = bottom - top + 0.5f;
    // float box_width = right - left;
    // float box_height = bottom - top;

    // float scale_to_predict_x = 0.25f;
    // float scale_to_predict_y = 0.25f;
    // int mask_out_width = box_width * scale_to_predict_x + 0.5f;
    // int mask_out_height = box_height * scale_to_predict_y + 0.5f;
    // int bytes_of_mask_out = mask_out_width * mask_out_height;
    // TRT::Tensor mask_out_device(TRT::DataType::UInt8);
    // mask_out_device.resize(1, 1*box_width*box_height).to_gpu();
    // uint8_t *mask_out_gpu = mask_out_device.gpu<uint8_t>();
    // decode_single_mask(left * scale_to_predict_x, top * scale_to_predict_y, feat_tensor,
    //                     base_tensor,
    //                     512, 512, mask_out_gpu,
    //                     784, mask_out_width, mask_out_height, stream_);
    
    // decode_single_mask(left, top, feat_tensor,
    //                     base_tensor,
    //                     512, 512, mask_out_gpu,
    //                     784, box_width, box_height, stream_);
    // uint8_t *mask_out_cpu = mask_out_device.cpu<uint8_t>();
    // cv::Mat masks_mat(box_width, box_height, CV_8UC1, mask_out_cpu);

    // cv::imwrite("./tmp.png", masks_mat);

    // printf("**********************");
    // int h = 3648;
    // int w = 5472;
    // int stride = 8;

    // TRT::Tensor locations_device(TRT::DataType::Float);

    // int h_step = static_cast<int>(std::ceil(static_cast<float>(h) / stride)); // 向上取整
    // int w_step = static_cast<int>(std::ceil(static_cast<float>(w) / stride)); // 向上取整
    // locations_device.resize(h_step*w_step, 2).to_gpu(); 
    
    // auto locations_ptr = locations_device.gpu<float>();

    // cudaEvent_t start, stop;
    // cudaEventCreate(&start);
    // cudaEventCreate(&stop);

    // cudaEventRecord(start);

    // compute_locations(h_step, w_step, stride, locations_ptr);

    // cudaEventRecord(stop);
    // cudaEventSynchronize(stop);

    // float milliseconds = 0;
    // cudaEventElapsedTime(&milliseconds, start, stop);

    // std::cout << "Kernel execution time: " << milliseconds << " ms" << std::endl;
    // auto locations_cpu = locations_device.cpu<float>();
    // printf("********************\n");
    // for(int i =0; i<15; ++i){
    //     printf("**%.01f-%.01f**\n", locations_cpu[i*2], locations_cpu[i*2+1]);
    // }
    return 0;
}