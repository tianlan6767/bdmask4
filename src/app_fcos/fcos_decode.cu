#include <common/cuda_tools.hpp>

namespace Fcos{



    const int NUM_BOX_ELEMENT = 792;      // left, top, right, bottom, confidence, class, keepflag, box_index
    __device__ const int FEATURE_STRIDES[] = {8, 16, 32, 64, 128};
    static __device__ void affine_project(float* matrix, float x, float y, float* ox, float* oy){
        *ox = matrix[0] * x + matrix[1] * y + matrix[2];
        *oy = matrix[3] * x + matrix[4] * y + matrix[5];
    }

    static __global__ void decode_kernel(float* predict, int num_bboxes, int num_classes, float confidence_threshold, 
                                        float* invert_affine_matrix, float* parray, int max_objects){  

        int position = blockDim.x * blockIdx.x + threadIdx.x;
        // printf("当前位置-%d\n", position);
		if (position >= num_bboxes) return;

        float* pitem     = predict + (5 + num_classes + 784) * position;
        float objectness = pitem[4];
        if(objectness < confidence_threshold)
            return;

        float* class_confidence = pitem + 5;
        float confidence        = *class_confidence++;
        int label               = 0;
        for(int i = 1; i < num_classes; ++i, ++class_confidence){
            if(*class_confidence > confidence){
                confidence = *class_confidence;
                label      = i;
            }
        }

        confidence = objectness;
        if(confidence < confidence_threshold)
            return;

        int index = atomicAdd(parray, 1);
        if(index >= max_objects)
            return;
        float left       = *pitem++;
        float top        = *pitem++;
        float right      = *pitem++;
        float bottom     = *pitem++;
        affine_project(invert_affine_matrix, left,  top,    &left,  &top);
        affine_project(invert_affine_matrix, right, bottom, &right, &bottom);

        float* pout_item = parray + 1 + index * NUM_BOX_ELEMENT;

        *pout_item++ = left;
        *pout_item++ = top;
        *pout_item++ = right;
        *pout_item++ = bottom;
        *pout_item++ = confidence;
        *pout_item++ = label;
        *pout_item++ = 1; // 1 = keep, 0 = ignore
        *pout_item++ = position;


        for(int i = 0; i < 784; ++i){
            *pout_item++ = *(pitem+i+1+num_classes);
        }
    }

    static __device__ float box_iou(
        float aleft, float atop, float aright, float abottom, 
        float bleft, float btop, float bright, float bbottom
    ){

        float cleft 	= max(aleft, bleft);
        float ctop 		= max(atop, btop);
        float cright 	= min(aright, bright);
        float cbottom 	= min(abottom, bbottom);
        
        float c_area = max(cright - cleft, 0.0f) * max(cbottom - ctop, 0.0f);
        if(c_area == 0.0f)
            return 0.0f;
        
        float a_area = max(0.0f, aright - aleft) * max(0.0f, abottom - atop);
        float b_area = max(0.0f, bright - bleft) * max(0.0f, bbottom - btop);
        return c_area / (a_area + b_area - c_area);
    }

    static __global__ void nms_kernel(float* bboxes, int max_objects, float threshold){

        int position = (blockDim.x * blockIdx.x + threadIdx.x);
        int count = min((int)*bboxes, max_objects);
        if (position >= count) 
            return;
        
        // left, top, right, bottom, confidence, class, keepflag, num_index
        float* pcurrent = bboxes + 1 + position * NUM_BOX_ELEMENT;
        for(int i = 0; i < count; ++i){
            float* pitem = bboxes + 1 + i * NUM_BOX_ELEMENT;
            if(i == position || pcurrent[5] != pitem[5]) continue;

            if(pitem[4] >= pcurrent[4]){
                if(pitem[4] == pcurrent[4] && i < position)
                    continue;

                float iou = box_iou(
                    pcurrent[0], pcurrent[1], pcurrent[2], pcurrent[3],
                    pitem[0],    pitem[1],    pitem[2],    pitem[3]
                );
                if(iou > threshold){
                    pcurrent[6] = 0;  // 1=keep, 0=ignore
                    return;
                }
            }
        }
    } 


    static __global__ void recount_box_kernel(float* bboxes, int h, int w, int max_objects){
        int position = blockDim.x * blockIdx.x + threadIdx.x;
        int count = min((int)*bboxes, max_objects);
        if (position >= count) 
            return;
        float* pcurrent = bboxes + 1 + position * NUM_BOX_ELEMENT;
        int keepflag = pcurrent[6];
        if (keepflag){
            int origin_box_index = pcurrent[7];
            int start_postion = 0;
            for(int n=0; n < (sizeof(FEATURE_STRIDES) / sizeof(FEATURE_STRIDES[0])); ++n){
                int stride = FEATURE_STRIDES[n];
                int h_step = static_cast<int>(ceil(static_cast<float>(h) / stride));
                int w_step = static_cast<int>(ceil(static_cast<float>(w) / stride));
                int feature_hw = h_step * w_step;
                int end_postion = start_postion + feature_hw;
                
                if (start_postion < origin_box_index && origin_box_index < end_postion){
                    int feature_posion = origin_box_index - start_postion;
                    int feature_x = feature_posion % w_step;
                    int feature_y = feature_posion / w_step;
                    float feature_cx = 0 + feature_x * stride + stride / 2.0f;
                    float feature_cy = 0 + feature_y * stride + stride / 2.0f;
                    pcurrent[0] = feature_cx - pcurrent[0];
                    pcurrent[1] = feature_cy - pcurrent[1];
                    pcurrent[2] = feature_cx + pcurrent[2];
                    pcurrent[3] = feature_cy + pcurrent[3];
                    break;
                }
                start_postion = end_postion;
            }
        }
    }

    static __device__ float bilinear_interpolate(const float* bottom_data,const int height,const int width,float y,float x, const int index /* index for debug only*/) {
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


    static __global__ void decode_interpolate_kernel(float* input, const int channels, const int input_height, const int input_width,
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


    static __global__ void softmax_kernel(float* input, const int height, const int width, int jobs) {
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

    static __global__ void mul_sum_sigmod_kernel(float* top_data, float*feat_out_tensor, int height, int width, float* mask_pred, int jobs) {
        
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


    static __global__ void RoIAlignForward(
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

    static __global__ void generate_grid_kernel(const float* box, int N, int h, int w, float* box_grid, int jobs) {
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

    static __global__ void decode_grid_sample_kernel(float* mask, int mask_height, int mask_width, float* box_grid,  int grid_h, int grid_w, uint8_t* box_mask, int jobs){
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= jobs) return;
        float grid_x = (mask_width / 2.0f) * ((box_grid[idx*2]) + 1) ;
        float grid_y = (mask_height / 2.0f) * (box_grid[idx*2 + 1] + 1);
        box_mask[idx] = (bilinear_interpolate(mask, mask_height, mask_width, grid_y, grid_x, idx)) > 0.5f ? 255 : 0;
    }

    void decode_grid_sample(float* mask, int mask_height, int mask_width, float* box_grid,  int grid_height, int grid_width, uint8_t* box_mask, cudaStream_t stream){
        int jobs   = grid_height * grid_width;
        auto grid  = CUDATools::grid_dims(jobs);
        auto block = CUDATools::block_dims(jobs);
        checkCudaKernel(decode_grid_sample_kernel<<<grid, block, 0, stream>>>(
                mask, mask_height, mask_width, box_grid, grid_height, grid_width, box_mask, jobs
        ));
    }

    void decode_softmax(float* input, int height, int width, cudaStream_t stream) {
        int jobs   = width * height;
        auto grid  = CUDATools::grid_dims(jobs);
        auto block = CUDATools::block_dims(jobs);
        checkCudaKernel(softmax_kernel<<<grid, block, 0, stream>>>(
            input, height, width, jobs
        ));
    }

    void generate_grid(float* box, int N, int height, int width, float* box_grid, cudaStream_t stream){
        int jobs   = N * width * height;
        auto grid  = CUDATools::grid_dims(jobs);
        auto block = CUDATools::block_dims(jobs);
        checkCudaKernel(generate_grid_kernel<<<grid, block, 0, stream>>>(
                box, N, height, width, box_grid, jobs
        ));
    }

    void decode_mul_sum_sigmod(float * top_data, float * feat_out_tensor, int height, int width,  float*mask_pred, cudaStream_t stream){
        int jobs   = width * height;
        auto grid  = CUDATools::grid_dims(jobs);
        auto block = CUDATools::block_dims(jobs);
        checkCudaKernel(mul_sum_sigmod_kernel<<<grid, block, 0, stream>>>(
                top_data, feat_out_tensor, height, width, mask_pred, jobs
        ));
    }

    void decode_roialign(float* bottom_data,float spatial_scale, int channels, int height, int width, int pooled_height, int pooled_width, float sampling_ratio, float* bottom_rois,
        float* top_data, bool aligned, cudaStream_t stream) {
        int jobs   = channels * pooled_height * pooled_width;
        auto grid  = CUDATools::grid_dims(jobs);   
        auto block = CUDATools::block_dims(jobs);
        checkCudaKernel(RoIAlignForward<<<grid, block, 0, stream>>>(
            bottom_data, spatial_scale, channels, height, width, pooled_height, pooled_width, sampling_ratio, bottom_rois,
            top_data, aligned));
    }

    void decode_interpolate(float* src, int channels, int src_height, int src_width, float* dst, int dst_height, int dst_width,float scale_factor,
            cudaStream_t stream) {
            int jobs   = channels * dst_width * dst_height;
            auto grid  = CUDATools::grid_dims(jobs);
            auto block = CUDATools::block_dims(jobs);
            checkCudaKernel(decode_interpolate_kernel<<<grid, block, 0, stream>>>(
                src, channels, src_height, src_width, dst, dst_height, dst_width, scale_factor, jobs));
        }

    void decode_kernel_invoker(float* predict, int num_bboxes, int num_classes, float confidence_threshold, float* invert_affine_matrix, float* parray, int max_objects, cudaStream_t stream){
        auto grid = CUDATools::grid_dims(num_bboxes);
        auto block = CUDATools::block_dims(num_bboxes);
        checkCudaKernel(decode_kernel<<<grid, block, 0, stream>>>(predict, num_bboxes, num_classes, confidence_threshold, invert_affine_matrix, parray, max_objects));
    }

    void nms_kernel_invoker(float* parray, float nms_threshold, int max_objects, cudaStream_t stream){
        
        auto grid = CUDATools::grid_dims(max_objects);
        auto block = CUDATools::block_dims(max_objects);
        checkCudaKernel(nms_kernel<<<grid, block, 0, stream>>>(parray, max_objects, nms_threshold));
    }

    void recount_box(float* parray, int h, int w, int max_objects, cudaStream_t stream){
        auto grid = CUDATools::grid_dims(max_objects);
        auto block = CUDATools::block_dims(max_objects);
        checkCudaKernel(recount_box_kernel<<<grid, block, 0, stream>>>(parray, h, w,max_objects));
    }
};