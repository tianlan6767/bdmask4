#include "bdm.hpp"
#include <atomic>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <infer/trt_infer.hpp>
#include <common/ilogger.hpp>
#include <common/infer_controller.hpp>
#include <common/preprocess_kernel.cuh>
#include <common/monopoly_allocator.hpp>
#include <common/cuda_tools.hpp>

namespace Bdm{
    using namespace cv;
    using namespace std;


    void decode_kernel_invoker(
        float* predict, int num_bboxes, int num_classes, float confidence_threshold, 
        float* invert_affine_matrix, float* parray,
        int max_objects, cudaStream_t stream
    );

    void nms_kernel_invoker(
        float* parray, float nms_threshold, int max_objects, cudaStream_t stream
    );

    void insert_value(float *array, int* indexes, int k, float data, int position)
    {
        
        //数值比最小的还小
        if (data < array[k - 1])
        {
            return;
        }

        // 19, 18, 17, 16,.........4, 3, 2, 1, 0
        for (int i = k - 2; i >= 0; i--)
        {
            if (data > array[i])
            {
                array[i + 1] = array[i];
                indexes[i + 1] = indexes[i];
            }
            else if(data == array[i]){
                for (int j = k - 2; j > i; j--)
                {
                    array[j] = array[j - 1];
                    indexes[j] = indexes[j - 1];
                }
                array[i] = data;
                indexes[i] = position;
                return;
            }
            else
            {
                array[i + 1] = data;
                indexes[i+1] = position;
                return;
            }
        }

        array[0] = data;
        indexes[0] = position;
    }


    void cpu_topk(float * & input, int* indexes, int length, int topk)
    {
        // float cpu_result[topk] = {0};
        float* cpu_result = new float[topk]();
        printf("**************************************************************\n");
        printf("当前值未修改为:*********************************\n");
        for(int j=0; j < topk; j++){
            printf("%.2f # %d \n", cpu_result[j], indexes[j]);
        }
        printf("当前值未修改*********************:\n");
        for (int i = 0; i < length; i++)
        {   
            float* pbox = input + 1 + i *  791;
            int keepflag = pbox[6];
            if (keepflag){
                insert_value(cpu_result, indexes, topk, pbox[4], i);
            }
        }
        printf("当前值修改为:*******************************\n");
        for(int j=0; j < topk; j++){
            printf("%.2f # %d \n", cpu_result[j], indexes[j]);
        }
        printf("\n");
        printf("**************************************************************\n");
        delete[] cpu_result;
    }



    struct AffineMatrix{
        float i2d[6];       // image to dst(network), 2x3 matrix
        float d2i[6];       // dst to image, 2x3 matrix

        void compute(const cv::Size& from, const cv::Size& to){
            float scale_x = to.width / (float)from.width;
            float scale_y = to.height / (float)from.height;
            float scale = std::min(scale_x, scale_y);

            i2d[0] = scale;  i2d[1] = 0;  i2d[2] = -scale * from.width  * 0.5  + to.width * 0.5 + scale * 0.5 - 0.5;
            i2d[3] = 0;  i2d[4] = scale;  i2d[5] = -scale * from.height * 0.5 + to.height * 0.5 + scale * 0.5 - 0.5;

            // 有了i2d矩阵，我们求其逆矩阵，即可得到d2i（用以解码时还原到原始图像分辨率上）
            cv::Mat m2x3_i2d(2, 3, CV_32F, i2d);
            cv::Mat m2x3_d2i(2, 3, CV_32F, d2i);
            cv::invertAffineTransform(m2x3_i2d, m2x3_d2i);
        }

        cv::Mat i2d_mat(){
            return cv::Mat(2, 3, CV_32F, i2d);
        }
    };

    using ControllerImpl = InferController
    <
        Mat,                    // input
        MaskArray,              // output
        tuple<string, string, int>,     // start param
        AffineMatrix            // additional
    >;
    class InferImpl : public Infer, public ControllerImpl{
    public:
        /** 要求在InferImpl里面执行stop，而不是在基类执行stop **/
        virtual ~InferImpl(){
            stop();
        }
        
        virtual bool startup(const string& fcos_file,const string& mask_file, int gpuid, float confidence_threshold, float mean[], float std[], float nms_threshold, NMSMethod nms_method){

            normalize_   = CUDAKernel::Norm::mean_std(mean, std, 1.0f);
            confidence_threshold_ = confidence_threshold;
            nms_threshold_        = nms_threshold;
            nms_method_           = nms_method;
            return ControllerImpl::startup(make_tuple(fcos_file, mask_file, gpuid));
        }

        virtual void worker(promise<bool>& result) override{

            string fcos_file = get<0>(start_param_);
            string mask_file = get<1>(start_param_);
            int gpuid   = get<2>(start_param_);

            TRT::set_device(gpuid);
            auto fcos_engine = TRT::load_infer(fcos_file);
            auto mask_engine = TRT::load_infer(mask_file);
            if(fcos_engine == nullptr || mask_engine == nullptr){
                INFOE("Engine %s load failed", fcos_file.c_str(), mask_file.c_str());
                result.set_value(false);
                return;
            }

            fcos_engine->print();
            mask_engine->print();

            const int MAX_IMAGE_BBOX = 1024;
            const int NUM_BOX_ELEMENT = 791;    // left, top, right, bottom, confidence, keepflag(1keep,0ignore), top_feat(784)
            const int MAX_KEEPFLAG    = 40;     //最多保留缺陷数量
            TRT::Tensor affin_matrix_device(TRT::DataType::Float);
            TRT::Tensor output_array_device(TRT::DataType::Float);
            TRT::Tensor box_feat_input_device(TRT::DataType::Float);
            TRT::Tensor box_output_device(TRT::DataType::Float);
            TRT::Tensor feature_output_device(TRT::DataType::UInt8);
            TRT::Tensor keepflag_indexes(TRT::DataType::Int32);



            int max_batch_size = fcos_engine->get_max_batch_size();
            auto input         = fcos_engine->input();
            auto output        = fcos_engine->output(1);
             auto bases_out     = fcos_engine->output(0);
            int num_classes    = output->size(2) - 789;
            
            input_width_       = input->size(3);
            input_height_      = input->size(2);

            auto base_input         = mask_engine->input(0);
            auto box_feat_input     = mask_engine->input(1);

            auto mask_output        = mask_engine->output();

            mask_input_width_       = base_input->size(3);
            mask_input_height_      = base_input->size(2);
            mask_feature_width_     = mask_output->size(2);
            mask_feature_height_    = mask_output->size(1);

            mask_box_feat_element_  = box_feat_input->size(1);
            

            tensor_allocator_  = make_shared<MonopolyAllocator<TRT::Tensor>>(max_batch_size * 2);
            stream_            = fcos_engine->get_stream();
            mask_engine->set_stream(stream_);
            gpu_               = gpuid;
            result.set_value(true);

            input->resize_single_dim(0, max_batch_size).to_gpu();


            affin_matrix_device.set_stream(stream_);

            // 这里8个值的目的是保证 8 * sizeof(float) % 32 == 0
            affin_matrix_device.resize(max_batch_size, 8).to_gpu();

            // 这里的 1 + MAX_IMAGE_BBOX结构是，counter + bboxes ...
            output_array_device.resize(max_batch_size, 1 + MAX_IMAGE_BBOX * NUM_BOX_ELEMENT).to_gpu(); 
            


            // mask参数
            box_output_device.resize(MAX_KEEPFLAG, 1 + 6).to_gpu();  // counter, left, right, top, bottom, conf, label
            feature_output_device.resize(MAX_KEEPFLAG, mask_output->size(1), mask_output->size(2)).to_gpu();
            box_feat_input_device.resize(MAX_KEEPFLAG, mask_box_feat_element_).to_gpu();
            keepflag_indexes.resize(max_batch_size, MAX_KEEPFLAG).to_gpu();

            vector<Job> fetch_jobs;
            int num = 1;
            while(get_jobs_and_wait(fetch_jobs, max_batch_size)){

                int infer_batch_size = fetch_jobs.size();
                input->resize_single_dim(0, infer_batch_size);

                for(int ibatch = 0; ibatch < infer_batch_size; ++ibatch){
                    auto& job  = fetch_jobs[ibatch];
                    auto& mono = job.mono_tensor->data();
                    affin_matrix_device.copy_from_gpu(affin_matrix_device.offset(ibatch), mono->get_workspace()->gpu(), 6);
                    input->copy_from_gpu(input->offset(ibatch), mono->gpu(), mono->count());
                    job.mono_tensor->release();
                }

                fcos_engine->forward(false);
                output_array_device.to_gpu(false);
                
                // 进行框处理和NMS计算
                for(int ibatch = 0; ibatch < infer_batch_size; ++ibatch){
                    auto& job                 = fetch_jobs[ibatch];
                    float* image_pred_output = output->gpu<float>(ibatch);
                    float* output_array_ptr   = output_array_device.gpu<float>(ibatch);
                    auto affine_matrix        = affin_matrix_device.gpu<float>(ibatch);
                    checkCudaRuntime(cudaMemsetAsync(output_array_ptr, 0, sizeof(float)*max_batch_size*(1 + MAX_IMAGE_BBOX * NUM_BOX_ELEMENT), stream_));
   
                    decode_kernel_invoker(image_pred_output, output->size(1), num_classes, confidence_threshold_, affine_matrix, output_array_ptr, MAX_IMAGE_BBOX, stream_);

                    if(nms_method_ == NMSMethod::FastGPU){
                        nms_kernel_invoker(output_array_ptr, nms_threshold_, MAX_IMAGE_BBOX, stream_);
                    }
                }
                output_array_device.save_to_file("./mask_data/tdata/out_array_" + to_string(num));
                num++;

                // mask推理
                // box_output_device.to_gpu(false);
                // box_feat_input_device.to_gpu(false);
                // feature_output_device.to_gpu(false);
                // keepflag_indexes.to_gpu(false);
                
                for(int ibatch = 0; ibatch < infer_batch_size; ++ibatch){
                    
                    float* box_feat_input_ptr = box_feat_input_device.gpu<float>(ibatch);
                    float* box_output_ptr = box_output_device.gpu<float>(ibatch);
                    uint8_t* feature_output_ptr = feature_output_device.gpu<uint8_t>(ibatch);
                    int32_t* keepflag_indexes_ptr = keepflag_indexes.gpu<int32_t>(ibatch);

                    checkCudaRuntime(cudaMemsetAsync(box_feat_input_ptr, 0, sizeof(float), stream_));
                    checkCudaRuntime(cudaMemsetAsync(box_output_ptr, 0, sizeof(float), stream_));
                    checkCudaRuntime(cudaMemsetAsync(feature_output_ptr, 0, sizeof(uint8_t), stream_));
                    checkCudaRuntime(cudaMemsetAsync(keepflag_indexes_ptr, -1, sizeof(uint8_t), stream_));

                    float* output_array_ptr   = output_array_device.gpu<float>(ibatch);

                    CUDAKernel::top_indexes(output_array_ptr, keepflag_indexes_ptr, MAX_IMAGE_BBOX, MAX_KEEPFLAG, stream_);
                    CUDAKernel::filter_box_feat_sorted(output_array_ptr, box_feat_input_ptr, box_output_ptr, keepflag_indexes_ptr, MAX_KEEPFLAG, stream_);

                    float* box_output_parray = box_output_device.cpu<float>(ibatch);   
                    int keep_box_counts  = min(MAX_KEEPFLAG, (int)*box_output_parray);
                    int32_t* keepflag_indexes_cpu = keepflag_indexes.cpu<int32_t>(ibatch);
                    
                    if (keep_box_counts > 0){

                        size_t size_image_feature =   iLogger::upbound(mask_input_width_*mask_input_height_ * 4 , 32);
                        size_t size_box_feat     =  iLogger::upbound(mask_box_feat_element_, 32);  

                        box_feat_input->resize_single_dim(0, keep_box_counts);
                        box_feat_input->resize_single_dim(1, size_box_feat);

                        base_input->copy_from_gpu(base_input->offset(ibatch), bases_out->gpu<float>(ibatch), size_image_feature);
                        box_feat_input->copy_from_gpu(box_feat_input->offset(ibatch), box_feat_input_device.gpu<float>(ibatch), keep_box_counts*mask_box_feat_element_);

                        mask_engine->forward(false);
                        feature_output_device.to_gpu(false);
                        float* mask_ptr = mask_output->gpu<float>(ibatch);

                        // mask阈值过滤
                        CUDAKernel::threshold_feature_mat(mask_ptr, feature_output_ptr, keep_box_counts, mask_feature_height_, mask_feature_width_, 0.5f, stream_);
                    }
                }
                
                // 后处理保存图片
                box_output_device.to_cpu();
                
                for(int ibatch = 0; ibatch < infer_batch_size; ++ibatch){
                    
                    auto & job       = fetch_jobs[ibatch];
                    auto & image_based_detects = job.output;
                    uint8_t* feature_output_ptr = feature_output_device.gpu<uint8_t>(ibatch);
                    float* box_output_parray = box_output_device.cpu<float>(ibatch);   
                    int keep_box_counts  = min(MAX_KEEPFLAG, (int)*box_output_parray);
                           
                    for(int i = 0; i < keep_box_counts; ++i){
                        Obj_mask obj_mask;
                        obj_mask.mask = Mat::zeros(mask_feature_height_, mask_feature_height_, CV_8UC1);
                        float* pbox  = box_output_parray + 1 + i * 6;
                        obj_mask.left       = pbox[0];
                        obj_mask.top        = pbox[1];
                        obj_mask.right      = pbox[2];
                        obj_mask.bottom     = pbox[3];
                        obj_mask.confidence = pbox[4];
                        obj_mask.class_label = pbox[5];
                        cudaMemcpy(obj_mask.mask.data, feature_output_ptr + i * mask_feature_height_ * mask_feature_width_*sizeof(uint8_t), sizeof(uint8_t)* mask_feature_height_* mask_feature_width_, cudaMemcpyDeviceToHost);
                        image_based_detects.emplace_back(obj_mask);
                    }

                    job.pro->set_value(image_based_detects);
                }
                fetch_jobs.clear();
            }
            stream_ = nullptr;
            tensor_allocator_.reset();
            INFOV("Engine destroy.");
        }
        

        virtual bool preprocess(Job& job, const Mat& image) override{
            int channel = image.channels();
            if(tensor_allocator_ == nullptr){
                INFOE("tensor_allocator_ is nullptr");
                return false;
            }

            job.mono_tensor = tensor_allocator_->query();
            if(job.mono_tensor == nullptr){
                INFOE("Tensor allocator query failed.");
                return false;
            }

            CUDATools::AutoDevice auto_device(gpu_);
            auto& tensor = job.mono_tensor->data();
            if(tensor == nullptr){
                // not init
                tensor = make_shared<TRT::Tensor>();
                tensor->set_workspace(make_shared<TRT::MixMemory>());
            }

            Size input_size(input_width_, input_height_);
            job.additional.compute(image.size(), input_size);
            
            tensor->set_stream(stream_);
            tensor->resize(1, channel, input_height_, input_width_);

            size_t size_image      = image.cols * image.rows * channel;
            size_t size_matrix     = iLogger::upbound(sizeof(job.additional.d2i), 32);
            auto workspace         = tensor->get_workspace();
            uint8_t* gpu_workspace        = (uint8_t*)workspace->gpu(size_matrix + size_image);
            float*   affine_matrix_device = (float*)gpu_workspace;
            uint8_t* image_device         = size_matrix + gpu_workspace;

            uint8_t* cpu_workspace        = (uint8_t*)workspace->cpu(size_matrix + size_image);
            float* affine_matrix_host     = (float*)cpu_workspace;
            uint8_t* image_host           = size_matrix + cpu_workspace;

            // speed up
            memcpy(image_host, image.data, size_image);
            memcpy(affine_matrix_host, job.additional.d2i, sizeof(job.additional.d2i));
            checkCudaRuntime(cudaMemcpyAsync(image_device, image_host, size_image, cudaMemcpyHostToDevice, stream_));
            checkCudaRuntime(cudaMemcpyAsync(affine_matrix_device, affine_matrix_host, sizeof(job.additional.d2i), cudaMemcpyHostToDevice, stream_));

            CUDAKernel::warp_affine_bilinear_and_normalize_plane(
                image_device,         image.cols * channel,       image.cols,       image.rows, 
                tensor->gpu<float>(), input_width_,         input_height_, 
                affine_matrix_device, 0, 
                normalize_, stream_
            );

            
            // tensor->save_to_file("/media/ps/data/train/LQ/LQ/bdmask/workspace/inf/1.jpg");

            return true;
        }

        virtual vector<shared_future<MaskArray>> commits(const vector<Mat>& images) override{
            return ControllerImpl::commits(images);
        }

        virtual std::shared_future<MaskArray> commit(const Mat& image) override{
            return ControllerImpl::commit(image);
        }

    private:
        int input_width_            = 0;
        int input_height_           = 0;
        int gpu_                    = 0;
        float confidence_threshold_ = 0;
        float nms_threshold_        = 0;

        int mask_input_width_            = 0;
        int mask_input_height_           = 0;
        int mask_feature_width_          = 0;
        int mask_feature_height_         = 0;
        int mask_num_batch_size_         = 0;
        int mask_box_feat_element_       = 0;
        float mask_confindence_          = 0.5f;

        NMSMethod nms_method_       = NMSMethod::FastGPU;
        TRT::CUStream stream_       = nullptr;
        CUDAKernel::Norm normalize_;
    };


    shared_ptr<Infer> create_infer(const string& fcos_file, const string& mask_file, int gpuid, float confidence_threshold,  float mean[], float std[], float nms_threshold, NMSMethod nms_method){
        shared_ptr<InferImpl> instance(new InferImpl());
        if(!instance->startup(fcos_file, mask_file, gpuid, confidence_threshold, mean, std,  nms_threshold, nms_method)){
            instance.reset();
        }
        return instance;
    }
};