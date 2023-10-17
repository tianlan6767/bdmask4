// #include <iostream>
// #include <vector>
// #include <cuda_runtime.h>
// #include <common/cuda_tools.hpp>
// #include <common/trt_tensor.hpp>

// // #define CUDA_1D_KERNEL_LOOP(i, n)                            \
// //   for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
// //        i += blockDim.x * gridDim.x)

// #define INTER_RESIZE_COEF_BITS 11
// #define INTER_RESIZE_COEF_SCALE (1 << INTER_RESIZE_COEF_BITS)
// #define CAST_BITS (INTER_RESIZE_COEF_BITS << 1)

// template<typename _T>
// static __inline__ __device__ _T limit(_T value, _T low, _T high){
//     return value < low ? low : (value > high ? high : value);
// }

// static __inline__ __device__ int resize_cast(int value){
//     return (value + (1 << (CAST_BITS - 1))) >> CAST_BITS;
// }

// __global__ void resize_bilinear_kernel(
//     float* src, int src_line_size, int src_width, int src_height, float* dst, int dst_width, int dst_height, 
//     float sx, float sy, int edge
// ){
//     int position = blockDim.x * blockIdx.x + threadIdx.x;
//     if (position >= edge) return;

//     int dx      = position % dst_width;
//     int dy      = position / dst_width;
//     float src_x = (dx + 0.5f) * sx - 0.5f;
//     float src_y = (dy + 0.5f) * sy - 0.5f;
//     float c0, c1, c2, c3;

//     int y_low = floorf(src_y);
//     int x_low = floorf(src_x);
//     int y_high = limit(y_low + 1, 0, src_height - 1);
//     int x_high = limit(x_low + 1, 0, src_width - 1);
//     y_low = limit(y_low, 0, src_height - 1);
//     x_low = limit(x_low, 0, src_width - 1);

//     int ly    = rint((src_y - y_low) * INTER_RESIZE_COEF_SCALE);
//     int lx    = rint((src_x - x_low) * INTER_RESIZE_COEF_SCALE);
//     int hy    = INTER_RESIZE_COEF_SCALE - ly;
//     int hx    = INTER_RESIZE_COEF_SCALE - lx;
//     int w1    = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;
//     float* v1 = src + y_low * src_line_size + x_low * 4;
//     float* v2 = src + y_low * src_line_size + x_high * 4;
//     float* v3 = src + y_high * src_line_size + x_low * 4;
//     float* v4 = src + y_high * src_line_size + x_high * 4;

//     c0 = resize_cast(w1 * v1[0] + w2 * v2[0] + w3 * v3[0] + w4 * v4[0]);
//     c1 = resize_cast(w1 * v1[1] + w2 * v2[1] + w3 * v3[1] + w4 * v4[1]);
//     c2 = resize_cast(w1 * v1[2] + w2 * v2[2] + w3 * v3[2] + w4 * v4[2]);
//     c3 = resize_cast(w1 * v1[3] + w2 * v2[3] + w3 * v3[3] + w4 * v4[3]);
//     auto max_val = fmaxf(fmaxf(c0, c1), fmaxf(c2, c3));
//     auto sum_exp = expf(c0) + expf(c1) + expf(c2)+ expf(c3);
//     int area = dst_width * dst_height;
//     float* pdst_c0 = dst + dy * dst_width + dx;
//     float* pdst_c1 = pdst_c0 + area;
//     float* pdst_c2 = pdst_c1 + area;
//     float* pdst_c3 = pdst_c2 + area;

//     *pdst_c0 = expf(c0 - max_val) / sum_exp;
//     *pdst_c1 = expf(c1 - max_val) / sum_exp;
//     *pdst_c2 = expf(c2 - max_val) / sum_exp;
//     *pdst_c3 = expf(c3 - max_val) / sum_exp;
// }



// template <typename T>
// __device__ T bilinear_interpolate(
//     const T* bottom_data,
//     const int height,
//     const int width,
//     T y,
//     T x,
//     const int index /* index for debug only*/) {
//   // deal with cases that inverse elements are out of feature map boundary
//   if (y < -1.0 || y > height || x < -1.0 || x > width) {
//     // empty
//     return 0;
//   }

//   if (y <= 0)
//     y = 0;
//   if (x <= 0)
//     x = 0;

//   int y_low = (int)y;
//   int x_low = (int)x;
//   int y_high;
//   int x_high;

//   if (y_low >= height - 1) {
//     y_high = y_low = height - 1;
//     y = (T)y_low;
//   } else {
//     y_high = y_low + 1;
//   }

//   if (x_low >= width - 1) {
//     x_high = x_low = width - 1;
//     x = (T)x_low;
//   } else {
//     x_high = x_low + 1;
//   }

//   T ly = y - y_low;
//   T lx = x - x_low;
//   T hy = 1. - ly, hx = 1. - lx;
//   // do bilinear interpolation
//   T v1 = bottom_data[y_low * width + x_low];
//   T v2 = bottom_data[y_low * width + x_high];
//   T v3 = bottom_data[y_high * width + x_low];
//   T v4 = bottom_data[y_high * width + x_high];
//   T w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

//   T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

//   return val;
// }


// template <typename T>
// __global__ void interpolate_kernel(T* input, const int channels, const int input_height, const int input_width,
//                                    T* output, const int output_height, const int output_width,
//                                    const float scale_factor, int jobs) {
//     int index = (blockIdx.x * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
//     if (index > jobs) return;
//     int output_y = (index / output_width) % output_height;
//     int output_x = index % output_width;
//     int c = (index / output_width / output_height) % channels;
//     int n = index / output_width / output_height / channels; 

//     T y = (T)((output_y + 0.5f) * scale_factor - 0.5f);
//     T x = (T)((output_x + 0.5f) * scale_factor - 0.5f);
//     // printf("**********************%d, %d, %d, %d\n", n, c, output_x, output_y);
//     const T* offset_bottom_data =
//         input + (n * channels + c) * input_height * input_width;
//     output[index] = bilinear_interpolate(offset_bottom_data, input_height, input_width, y, x, index);
// }

// void Interpolate(float* src, int channels, int src_height, int src_width, float* dst, int dst_height, int dst_width,float scale_factor,
//         cudaStream_t stream) {
//         int jobs   = channels * dst_width * dst_height;
//         auto grid  = CUDATools::grid_dims(jobs);
//         auto block = CUDATools::block_dims(jobs);
//         checkCudaKernel(interpolate_kernel<<<grid, block, 0, stream>>>(
//             src, channels, src_height, src_width, dst, dst_height, dst_width, scale_factor, jobs));
//     }


// __global__ void softmax_kernel(float* input, const int height, const int width, int jobs) {
//     int index = (blockIdx.x * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
//     if (index > jobs) return;
//     int dy = index / width;
//     int dx = index % width;
//     int area = height * width;
//     float* p_c0 = input + dy * width + dx;
//     float* p_c1 = p_c0 + area;
//     float* p_c2 = p_c1 + area;
//     float* p_c3 = p_c2 + area;
//     float c0 = *p_c0;
//     float c1 = *p_c1;
//     float c2 = *p_c2;
//     float c3 = *p_c3;
//     float max_val = fmaxf(fmaxf(c0, c1), fmaxf(c2, c3));
//     auto sum_exp = expf(c0) + expf(c1) + expf(c2)+ expf(c3);

//     *p_c0 = expf(c0 - max_val) / sum_exp;
//     *p_c1 = expf(c1 - max_val) / sum_exp;
//     *p_c2 = expf(c2 - max_val) / sum_exp;
//     *p_c3 = expf(c3 - max_val) / sum_exp;
// }

// void decode_softmax(float* input, int height, int width, cudaStream_t stream) {
//     int jobs   = width * height;
//     auto grid  = CUDATools::grid_dims(jobs);
//     auto block = CUDATools::block_dims(jobs);
//     checkCudaKernel(softmax_kernel<<<grid, block, 0, stream>>>(
//         input, height, width, jobs
//     ));
// }


// __global__ void mul_sum_sigmod_kernel(float* top_data, float*feat_out_tensor, int height, int width, float* mask_pred, int jobs) {
    
//     int index = (blockIdx.x * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
//     if (index > jobs) return;
//     int dy = index / width;
//     int dx = index % width;
//     int area = height * width;
//     float* p_c0 = top_data + dy * width + dx;
//     float* p_c1 = p_c0 + area;
//     float* p_c2 = p_c1 + area;
//     float* p_c3 = p_c2 + area;
//     float* f_c0 = feat_out_tensor + dy * width + dx;
//     float* f_c1 = f_c0 + area;
//     float* f_c2 = f_c1 + area;
//     float* f_c3 = f_c2 + area;
//     float* dst_mask = mask_pred + dy * width + dx;
//     *dst_mask = (*p_c0) * (*f_c0) + (*p_c1) * (*f_c1) + (*p_c2) * (*f_c2)+ (*p_c3) * (*f_c3);
// }


// template <typename T>
// __global__ void RoIAlignForward(
//     const T* bottom_data,
//     const T spatial_scale,
//     const int channels,
//     const int height,
//     const int width,
//     const int pooled_height,
//     const int pooled_width,
//     const int sampling_ratio,
//     const T* bottom_rois,
//     T* top_data,
//     bool aligned) {
//     // (n, c, ph, pw) is an element in the pooled output
//     int index = (blockIdx.x * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
//     int pw = index % pooled_width;
//     int ph = (index / pooled_width) % pooled_height;
//     int c = (index / pooled_width / pooled_height) % channels;
//     int n = index / pooled_width / pooled_height / channels;

//     const T* offset_bottom_rois = bottom_rois + n * 5;
//     int roi_batch_ind = offset_bottom_rois[0];

//     // Do not use rounding; this implementation detail is critical
//     T offset = aligned ? (T)0.5 : (T)0.0;
//     T roi_start_w = offset_bottom_rois[1] * spatial_scale - offset;
//     T roi_start_h = offset_bottom_rois[2] * spatial_scale - offset;
//     T roi_end_w = offset_bottom_rois[3] * spatial_scale - offset;
//     T roi_end_h = offset_bottom_rois[4] * spatial_scale - offset;

//     T roi_width = roi_end_w - roi_start_w;
//     T roi_height = roi_end_h - roi_start_h;
//     if (!aligned) { // for backward-compatibility only
//       roi_width = max(roi_width, (T)1.);
//       roi_height = max(roi_height, (T)1.);
//     }
//     T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
//     T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);

//     const T* offset_bottom_data =
//         bottom_data + (roi_batch_ind * channels + c) * height * width;

//     // We use roi_bin_grid to sample the grid and mimic integral
//     int roi_bin_grid_h = (sampling_ratio > 0)
//         ? sampling_ratio
//         : ceil(roi_height / pooled_height); // e.g., = 2
//     int roi_bin_grid_w =
//         (sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / pooled_width);

//     // We do average (integral) pooling inside a bin
//     // When the grid is empty, output zeros == 0/1, instead of NaN.
//     const T count = max(roi_bin_grid_h * roi_bin_grid_w, 1); // e.g. = 4

//     T output_val = 0.;
//     for (int iy = 0; iy < roi_bin_grid_h; iy++) // e.g., iy = 0, 1
//     {
//       const T y = roi_start_h + ph * bin_size_h +
//           static_cast<T>(iy + .5f) * bin_size_h /
//               static_cast<T>(roi_bin_grid_h); // e.g., 0.5, 1.5
//       for (int ix = 0; ix < roi_bin_grid_w; ix++) {
//         const T x = roi_start_w + pw * bin_size_w +
//             static_cast<T>(ix + .5f) * bin_size_w /
//                 static_cast<T>(roi_bin_grid_w);
//         // printf("%f, %f, %f, %f, %f, %f, %f, %f, %f, %f\n", roi_height, roi_width, roi_start_h, roi_start_w, roi_bin_grid_h, roi_bin_grid_w, bin_size_h, bin_size_w, x, y);
//         T val = bilinear_interpolate(
//             offset_bottom_data, height, width, y, x, index);
//         output_val += val;
//       }
//     }
//     output_val /= count;
//     top_data[index] = output_val;
//   }


// // static __global__ void decode_single_mask_kernel(int left, int top, float *mask_weights,
// //                                                  float *mask_predict, int mask_width,
// //                                                  int mask_height, unsigned char *mask_out,
// //                                                  int mask_dim, int out_width, int out_height) {
// //   // mask_predict to mask_out
// //   // mask_weights @ mask_predict
// //   int dx = blockDim.x * blockIdx.x + threadIdx.x;
// //   int dy = blockDim.y * blockIdx.y + threadIdx.y;
// //   if (dx >= out_width || dy >= out_height) return;

// //   int sx = left + dx;
// //   int sy = top + dy;
// //   printf("8************************%d, %d %d, %d\n", sx, sy, mask_width, mask_height);
// //   if (sx < 0 || sx >= mask_width || sy < 0 || sy >= mask_height) {
// //     mask_out[dy * out_width + dx] = 0;
// //     return;
// //   }
// //   printf("tttt**********************%d, %d %d, %d\n", sx, sy, mask_width, mask_height);
// //   float cumprod = 0;
// //   for (int ic = 0; ic < mask_dim; ++ic) {
// //     printf("mask_dim:%d\n", mask_dim);
// //     float cval = mask_predict[(ic * mask_height + sy) * mask_width + sx];
// //     float wval = mask_weights[ic];
// //     cumprod += cval * wval;
// //   }

// //   float alpha = 1.0f / (1.0f + exp(-cumprod));
// //   printf("-----------------%f\n", alpha);
// //   mask_out[dy * out_width + dx] = alpha * 255;
// // }

// // static void decode_single_mask(float left, float top, float *mask_weights, float *mask_predict,
// //                                int mask_width, int mask_height, unsigned char *mask_out,
// //                                int mask_dim, int out_width, int out_height, cudaStream_t stream) {
// //   // mask_weights is mask_dim(32 element) gpu pointer
// //   dim3 grid((out_width + 31) / 32, (out_height + 31) / 32);
// //   dim3 block(32, 32);

// //   checkCudaKernel(decode_single_mask_kernel<<<grid, block, 0, stream>>>(
// //       left, top, mask_weights, mask_predict, mask_width, mask_height, mask_out, mask_dim, out_width,
// //       out_height));
// // }




// void resize_bilinear(
// 		float* src, int src_line_size, int src_width, int src_height, float* dst, int dst_width, int dst_height,
// 		cudaStream_t stream) {
		
// 		int jobs   = dst_width * dst_height;
// 		auto grid  = CUDATools::grid_dims(jobs);
// 		auto block = CUDATools::block_dims(jobs);
		
// 		checkCudaKernel(resize_bilinear_kernel << <grid, block, 0, stream >> > (
// 			src, src_line_size,
// 			src_width, src_height, dst,
// 			dst_width, dst_height, src_width/(float)dst_width, src_height/(float)dst_height, jobs
// 		));
// 	}

// __global__ void generate_grid_kernel(const float* box, int N, int w, int h, float* grid, int jobs) {
//     int idx = threadIdx.x + blockIdx.x * blockDim.x;
//     if (idx > jobs) return;
//     int stride = blockDim.x * gridDim.x;

//     for (int i = idx; i < N * h * w; i += stride) {
//         int n = i / (h * w);
//         int y = (i % (h * w)) / w;
//         int x = (i % (h * w)) % w;

//         float img_y = y + int(box[n * 5 + 1]) + 0.5f;
//         float img_x = x + int(box[n * 5 + 2]) + 0.5f;
//         printf("%d, %d, %f, %f, %f, %f, %f, %f, %f\n", y, x, img_y, img_x, box[n * 5], box[n * 5 + 1], box[n * 5 + 2],box[n * 5 + 3],box[n * 5 + 4]);
//         float img_y_tmp = (img_y - box[n * 5 + 2]) / (box[n * 5 + 4] - box[n * 5 + 2]) * 2.0f - 1.0f;
//         float img_x_tmp = (img_x - box[n * 5 + 1]) / (box[n * 5 + 3] - box[n * 5 + 1]) * 2.0f - 1.0f;
//         printf("%f-%f-%f-%f    %f-%f-%f-%f-%f\n", img_y_tmp, img_y, box[n * 5 + 2], box[n * 5 + 4], img_x_tmp, img_x,box[n * 5 + 1], box[n * 5 + 3]);

//         grid[i * 2] = img_x;
//         grid[i * 2 + 1] = img_y;
//     }
// }

// static void generate_grid(float* box, int N, int height, int width, float* box_grid, cudaStream_t stream){
//     int jobs   = N * width * height;
//     auto grid  = CUDATools::grid_dims(jobs);
//     auto block = CUDATools::block_dims(jobs);
//     checkCudaKernel(generate_grid_kernel<<<grid, block, 0, stream>>>(
//             box, N, height, width, box_grid, jobs
//     ));
// }


// static void decode_mul_sum_sigmod(float * top_data, float * feat_out_tensor, int height, int width,  float*mask_pred, cudaStream_t stream){
//     int jobs   = width * height;
//     auto grid  = CUDATools::grid_dims(jobs);
//     auto block = CUDATools::block_dims(jobs);
//     checkCudaKernel(mul_sum_sigmod_kernel<<<grid, block, 0, stream>>>(
//             top_data, feat_out_tensor, height, width, mask_pred, jobs
//     ));
// }

// static void decode_roialign(float* bottom_data,float spatial_scale, int channels, int height, int width, int pooled_height, int pooled_width, float sampling_ratio, float* bottom_rois,
//       float* top_data, bool aligned, cudaStream_t stream) {
//     int jobs   = channels * pooled_height * pooled_width;
//     auto grid  = CUDATools::grid_dims(jobs);   // 8192
//     auto block = CUDATools::block_dims(jobs); // 512

//     checkCudaKernel(RoIAlignForward<<<grid, block, 0, stream>>>(
//         bottom_data, spatial_scale, channels, height, width, pooled_height, pooled_width, sampling_ratio, bottom_rois,
//         top_data, aligned));
// }




// int main() {
//     // 初始化 CUDA
//     cudaStream_t stream_ = nullptr;
//     cudaSetDevice(3);
//     // 定义时间戳变量
//     cudaEvent_t start, stop;
//     cudaEventCreate(&start);
//     cudaEventCreate(&stop);
//     TRT::Tensor base_device(TRT::DataType::Float);
//     TRT::Tensor box_device(TRT::DataType::Float);
//     TRT::Tensor feat_device(TRT::DataType::Float);
//     TRT::Tensor feat_out_device(TRT::DataType::Float);
//     TRT::Tensor top_device(TRT::DataType::Float);
//     TRT::Tensor mask_pred_device(TRT::DataType::Float);
//     TRT::Tensor box_grid_device(TRT::DataType::Float);
//     base_device.load_from_file("/media/ps/data/train/LQ/LQ/bdms/bdmask/workspace/models/JT/inf/base_input");
//     box_device.load_from_file("/media/ps/data/train/LQ/LQ/bdms/bdmask/workspace/models/JT/inf/box_input");
//     feat_device.load_from_file("/media/ps/data/train/LQ/LQ/bdms/bdmask/workspace/models/JT/inf/top_feat_input");
    
//     top_device.resize(1,4,56,56);
//     feat_device.resize(1,4,14,14);
//     feat_out_device.resize(1, 4, 56, 56);
//     mask_pred_device.resize(1, 56, 56);

//     float* box_cpu = box_device.cpu<float>();
//     int box_mask_height = box_cpu[4] - box_cpu[2] + 0.5f;
//     int box_mask_width  = box_cpu[3] - box_cpu[1] + 0.5f;
//     box_grid_device.resize(1, box_mask_height, box_mask_width, 2);


//     float* box_tensor = box_device.gpu<float>();
//     float* feat_tensor = feat_device.gpu<float>();
//     float* feat_out_tensor = feat_out_device.gpu<float>();
//     float* base_tensor = base_device.gpu<float>();
//     float* top_data = top_device.gpu<float>();
//     float* mask_pred = mask_pred_device.gpu<float>();
//     float* box_grid = box_grid_device.gpu<float>();



//     int channels = base_device.size(1);
//     int base_height = base_device.size(2);
//     int base_width = base_device.size(3);

//     float spatial_scale = 0.25f;
//     int pooled_height = 56;
//     int pooled_width = 56;
//     int sampling_ratio = 1;
//     bool aligned = true;

//     // 启动计时
//     cudaEventRecord(start, stream_);
//     // 调用 CUDA 核函数   
//     generate_grid(box_tensor, 1, box_mask_height, box_mask_width, box_grid, stream_);
//     decode_roialign(base_tensor, spatial_scale, channels, base_height, base_width, pooled_height, pooled_width, sampling_ratio, box_tensor, top_data, aligned, stream_);
//     Interpolate(feat_tensor, 4, 14, 14, feat_out_tensor, 56, 56, 0.25, stream_);
//     decode_softmax(feat_out_tensor, 56, 56, stream_);
//     decode_mul_sum_sigmod(top_data, feat_out_tensor, 56, 56,  mask_pred, stream_);
    
//     // 停止计时
//     cudaEventRecord(stop, stream_);
//     cudaEventSynchronize(stop);

//     // 计算执行时间
//     float elapsedTime;
//     cudaEventElapsedTime(&elapsedTime, start, stop);

//     top_device.save_to_file("/media/ps/data/train/LQ/LQ/bdms/bdmask/workspace/models/JT/inf/top_input-roialigned");
//     feat_device.save_to_file("/media/ps/data/train/LQ/LQ/bdms/bdmask/workspace/models/JT/inf/feat_resize");
//     feat_out_device.save_to_file("/media/ps/data/train/LQ/LQ/bdms/bdmask/workspace/models/JT/inf/feat_out_device");
//     mask_pred_device.save_to_file("/media/ps/data/train/LQ/LQ/bdms/bdmask/workspace/models/JT/inf/mask_pred_device");
//     box_grid_device.save_to_file("/media/ps/data/train/LQ/LQ/bdms/bdmask/workspace/models/JT/inf/box_grid_device");
//     // 打印执行时间
//     std::cout << "CUDA Kernel Execution Time: " << elapsedTime << " ms" << std::endl;

//     // 释放时间戳资源
//     cudaEventDestroy(start);
//     cudaEventDestroy(stop);
//     // float left, top, right, bottom;
//     // left = box_tensor[1];
//     // top = box_tensor[2];
//     // right = box_tensor[3];
//     // bottom = box_tensor[4];
//     // int box_width = right - left + 0.5f;
//     // int box_height = bottom - top + 0.5f;


//     // float box_width = right - left;
//     // float box_height = bottom - top;

//     // float scale_to_predict_x = 0.25f;
//     // float scale_to_predict_y = 0.25f;
//     // int mask_out_width = box_width * scale_to_predict_x + 0.5f;
//     // int mask_out_height = box_height * scale_to_predict_y + 0.5f;
//     // int bytes_of_mask_out = mask_out_width * mask_out_height;
//     // TRT::Tensor mask_out_device(TRT::DataType::UInt8);
//     // mask_out_device.resize(1, 1*box_width*box_height).to_gpu();

//     // uint8_t *mask_out_gpu = mask_out_device.gpu<uint8_t>();



//     // decode_single_mask(left * scale_to_predict_x, top * scale_to_predict_y, feat_tensor,
//     //                     base_tensor,
//     //                     512, 512, mask_out_gpu,
//     //                     784, mask_out_width, mask_out_height, stream_);
    
//     // decode_single_mask(left, top, feat_tensor,
//     //                     base_tensor,
//     //                     512, 512, mask_out_gpu,
//     //                     784, box_width, box_height, stream_);
//     // uint8_t *mask_out_cpu = mask_out_device.cpu<uint8_t>();
//     // cv::Mat masks_mat(box_width, box_height, CV_8UC1, mask_out_cpu);

//     // cv::imwrite("./tmp.png", masks_mat);

//     // printf("**********************");
//     // int h = 3648;
//     // int w = 5472;
//     // int stride = 8;

//     // TRT::Tensor locations_device(TRT::DataType::Float);

//     // int h_step = static_cast<int>(std::ceil(static_cast<float>(h) / stride)); // 向上取整
//     // int w_step = static_cast<int>(std::ceil(static_cast<float>(w) / stride)); // 向上取整
//     // locations_device.resize(h_step*w_step, 2).to_gpu(); 
    
//     // auto locations_ptr = locations_device.gpu<float>();

//     // cudaEvent_t start, stop;
//     // cudaEventCreate(&start);
//     // cudaEventCreate(&stop);

//     // cudaEventRecord(start);

//     // compute_locations(h_step, w_step, stride, locations_ptr);

//     // cudaEventRecord(stop);
//     // cudaEventSynchronize(stop);

//     // float milliseconds = 0;
//     // cudaEventElapsedTime(&milliseconds, start, stop);

//     // std::cout << "Kernel execution time: " << milliseconds << " ms" << std::endl;
//     // auto locations_cpu = locations_device.cpu<float>();
//     // printf("********************\n");
//     // for(int i =0; i<15; ++i){
//     //     printf("**%.01f-%.01f**\n", locations_cpu[i*2], locations_cpu[i*2+1]);
//     // }
//     return 0;
// }