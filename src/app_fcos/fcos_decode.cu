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