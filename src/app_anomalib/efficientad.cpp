#include "efficientad.hpp"
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

namespace EfficientAd{
    using namespace cv;
    using namespace std;

    struct AffineMatrix{
        float i2d[6];  // image to dst(network),  2*3 matrix
        float d2i[6];  // dst to image, 2*3 matrix
        void compute(const cv::Size& from , const cv::Size& to){
            float scale_x = to.width / (float)from.width;
            float scale_y = to.height / (float)from.height;
            float scale = std::min(scale_x, scale_y);

            i2d[0] = scale; i2d[1] = 0;     i2d[2] = -scale * from.width  * 0.5 + to.width  * 0.5 + scale * 0.5 - 0.5;
            i2d[3] = 0;     i2d[4] = scale; i2d[5] = -scale * from.height * 0.5 + to.height * 0.5 + scale * 0.5 - 0.5;

            cv::Mat m2x3_i2d(2, 3, CV_32F, i2d);
            cv::Mat m2x3_d2i(2, 3, CV_32F, d2i);
            cv::invertAffineTransform(m2x3_i2d, m2x3_d2i);
        }

        cv::Mat i2d_mat(){
            return cv::Mat(2, 3, CV_32F, i2d);
        }

    };


    void decode_kernel_invoker(
        float* predict, int height, int width, float confidence_threshold, uint8_t* parray, cudaStream_t stream
    );

    // void warp

    using ControllerImpl = InferController
    <
        Mat,                    // input
        BoxArray,               // output
        tuple<string, int>,     // start param
        AffineMatrix            // additional
    >;
    class InferImpl : public Infer, public ControllerImpl{
    public:
        /** 要求在InferImpl里面执行stop，而不是在基类执行stop **/
        virtual ~InferImpl(){
            stop();
        }
        
        virtual bool startup(const string& file, int gpuid, float confidence_threshold){

            normalize_   = CUDAKernel::Norm::alpha_beta(1 / 255.0f, 0.0f, CUDAKernel::ChannelType::Invert);
            // normalize_   = CUDAKernel::Norm::alpha_beta(1.0f, 0.0f, CUDAKernel::ChannelType::Invert);
            confidence_threshold_ = confidence_threshold;
            return ControllerImpl::startup(make_tuple(file, gpuid));
        }

        virtual void worker(promise<bool>& result) override{

            string file = get<0>(start_param_);
            int gpuid   = get<1>(start_param_);

            TRT::set_device(gpuid);
            auto engine = TRT::load_infer(file);
            if(engine == nullptr){
                INFOE("Engine %s load failed", file.c_str());
                result.set_value(false);
                return;
            }

            engine->print();

            TRT::Tensor affine_matrix_device(TRT::DataType::Float);
            TRT::Tensor affine_matrix_device_i2d(TRT::DataType::Float);
            TRT::Tensor output_array_device(TRT::DataType::UInt8);
            TRT::Tensor mask_array_device(TRT::DataType::UInt8);

            int max_batch_size = 1;
            auto input         = engine->input();
            auto output        = engine->output(0);
            output_height_ = output->size(2);
            output_width_ = output->size(3);
            input_width_       = input->size(3);
            input_height_      = input->size(2);
            tensor_allocator_  = make_shared<MonopolyAllocator<TRT::Tensor>>(max_batch_size * 2);
            stream_            = engine->get_stream();
            gpu_               = gpuid;
            result.set_value(true);

            input->resize_single_dim(0, max_batch_size).to_gpu();
            affine_matrix_device.set_stream(stream_);
            affine_matrix_device_i2d.set_stream(stream_);

            // 这里8个值的目的是保证 8 * sizeof(float) % 32 == 0
            affine_matrix_device.resize(max_batch_size, 8).to_gpu();
            affine_matrix_device_i2d.resize(max_batch_size, 8).to_gpu();

            // 这里的 1 + MAX_IMAGE_BBOX结构是，counter + bboxes ...
            output_array_device.resize(max_batch_size, output_height_, output_width_).to_gpu(); 
            mask_array_device.resize(max_batch_size, output_height_, output_width_).to_gpu(); 
            vector<Job> fetch_jobs;
            while(get_jobs_and_wait(fetch_jobs, max_batch_size)){

                int infer_batch_size = fetch_jobs.size();
                input->resize_single_dim(0, infer_batch_size);
                input->resize_single_dim(2, input_height_);
                input->resize_single_dim(3, input_width_);

                output->resize_single_dim(0, infer_batch_size);
                affine_matrix_device.resize_single_dim(0, infer_batch_size);

                affine_matrix_device_i2d.resize_single_dim(0, infer_batch_size);
                                
                mask_array_device.resize_single_dim(0, infer_batch_size);
                mask_array_device.resize_single_dim(1, orig_inp_height_);
                mask_array_device.resize_single_dim(2, orig_inp_width_);



                for(int ibatch = 0; ibatch < infer_batch_size; ++ibatch){
                    auto& job  = fetch_jobs[ibatch];
                    auto& mono = job.mono_tensor->data();
                    affine_matrix_device.copy_from_gpu(affine_matrix_device.offset(ibatch), mono->get_workspace()->gpu(), 6);
                    input->copy_from_gpu(input->offset(ibatch), mono->gpu(), mono->count());
                    job.mono_tensor->release();
                }

                engine->forward(false);
                output_array_device.to_gpu(false);
                
                for(int ibatch = 0; ibatch < infer_batch_size; ++ibatch){
                    auto& job                 = fetch_jobs[ibatch];
                    float* image_pred_output = output->gpu<float>(ibatch);
                    uint8_t* output_array_ptr   = output_array_device.gpu<uint8_t>(ibatch);
                    uint8_t* mask_array_ptr   = mask_array_device.gpu<uint8_t>(ibatch);
                    auto affine_matrix_i2d        = affine_matrix_device_i2d.gpu<float>(ibatch);
                    auto affine_matrix_cpu_i2d        = affine_matrix_device_i2d.cpu<float>(ibatch);
                    memcpy(affine_matrix_cpu_i2d, job.additional.i2d, sizeof(job.additional.i2d));
                    checkCudaRuntime(cudaMemcpyAsync(affine_matrix_i2d, affine_matrix_cpu_i2d, sizeof(job.additional.i2d), cudaMemcpyHostToDevice, stream_));
                    checkCudaRuntime(cudaMemsetAsync(output_array_ptr, 0, sizeof(uint8_t)*(output_height_ * output_width_), stream_));
                    checkCudaRuntime(cudaMemsetAsync(mask_array_ptr, 0, sizeof(uint8_t)*(orig_inp_height_ * orig_inp_width_), stream_));
                    
                    // clip (0, 1) * 255--》threshold 100
                    decode_kernel_invoker(image_pred_output, output_height_, output_width_, confidence_threshold_, output_array_ptr, stream_);


                    CUDAKernel::warp_affine_bilinear_mask(
                        output_array_ptr,         output_width_ * 1,       output_width_,       output_height_, 
                        mask_array_ptr,        orig_inp_height_,         orig_inp_width_, 
                        affine_matrix_i2d, 0, stream_);
                }

                output_array_device.to_cpu();
                mask_array_device.to_cpu();
                for(int ibatch = 0; ibatch < infer_batch_size; ++ibatch){
                    auto& job     = fetch_jobs[ibatch];
                    uint8_t* output_array_ptr   = output_array_device.cpu<uint8_t>(ibatch);
                    uint8_t* mask_array_ptr   = mask_array_device.cpu<uint8_t>(ibatch);

                    auto& image_based_boxes   = job.output;
                    Box result_object_box;
                    result_object_box.seg = make_shared<InstanceSegmentMap>(orig_inp_width_, orig_inp_height_);
                    result_object_box.seg->height = orig_inp_height_;
                    result_object_box.seg->width = orig_inp_width_;
                    uint8_t* mask_out_host = result_object_box.seg->data;
                    checkCudaRuntime(cudaMemcpyAsync(mask_out_host, mask_array_ptr, orig_inp_height_ * orig_inp_width_ * sizeof(uint8_t), cudaMemcpyDeviceToHost, stream_));

                    // result_object_box.seg = make_shared<InstanceSegmentMap>(output_width_, output_height_);
                    // result_object_box.seg->height = output_height_;
                    // result_object_box.seg->width = output_width_;
                    // uint8_t* mask_out_host = result_object_box.seg->data;
                    // checkCudaRuntime(cudaMemcpyAsync(mask_out_host, output_array_ptr, output_width_ * output_height_ * sizeof(uint8_t), cudaMemcpyDeviceToHost, stream_));
                    
                    image_based_boxes.emplace_back(result_object_box);
                    job.pro->set_value(image_based_boxes);
                }
                fetch_jobs.clear();
            }
            stream_ = nullptr;
            tensor_allocator_.reset();
            INFOV("Engine destroy.");
        }

        virtual bool preprocess(Job& job, const Mat& image) override{
            int channel = image.channels();
            orig_inp_width_ = image.cols;
            orig_inp_height_ = image.rows;
            if(tensor_allocator_ == nullptr){
                INFOE("tensor_allocator_ is nullptr");
                return false;
            }
            
            if(image.empty()){
                INFOE("Image is empty");
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

            tensor->set_stream(stream_);
            tensor->resize(1, channel, input_height_, input_width_);
            cv::Size image_size = image.size();
            Size input_size(input_width_, input_height_);
            job.additional.compute(image_size, input_size);

            size_t size_image      = image.cols * image.rows * channel;
            size_t size_matrix     = iLogger::upbound(sizeof(job.additional.d2i), 32);
            // auto workspace         = tensor->get_workspace();
            // uint8_t* gpu_workspace        = (uint8_t*)workspace->gpu(size_matrix * 2 + size_image);
            // float*   affine_matrix_device = (float*)gpu_workspace;
            // float*   affine_matrix_device_i2d = (float*)gpu_workspace + size_matrix;
            // uint8_t* image_device         = size_matrix * 2 + gpu_workspace;

            // uint8_t* cpu_workspace        = (uint8_t*)workspace->cpu(size_matrix * 2 + size_image);
            // float* affine_matrix_host     = (float*)cpu_workspace;
            // float* affine_matrix_host_i2d     = (float*)cpu_workspace + size_matrix;
            // uint8_t* image_host           = size_matrix * 2 + cpu_workspace;


            auto workspace         = tensor->get_workspace();
            uint8_t* gpu_workspace        = (uint8_t*)workspace->gpu(size_matrix  + size_image);
            float*   affine_matrix_device = (float*)gpu_workspace;
            uint8_t* image_device         = size_matrix  + gpu_workspace;

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
            return true;
        }

        virtual vector<shared_future<BoxArray>> commits(const vector<Mat>& images) override{
            return ControllerImpl::commits(images);
        }

        virtual std::shared_future<BoxArray> commit(const Mat& image) override{
            return ControllerImpl::commit(image);
        }

    private:
        int input_width_            = 0;
        int input_height_           = 0;
        int orig_inp_width_         = 0;
        int orig_inp_height_        = 0;
        int output_width_           = 0;
        int output_height_          = 0;
        int gpu_                    = 0;
        float confidence_threshold_ = 0;
        float nms_threshold_        = 0;
        NMSMethod nms_method_       = NMSMethod::FastGPU;
        TRT::CUStream stream_       = nullptr;
        CUDAKernel::Norm normalize_;
    };


    shared_ptr<Infer> create_infer(const string& engine_file, int gpuid, float confidence_threshold){
        shared_ptr<InferImpl> instance(new InferImpl());
        if(!instance->startup(engine_file, gpuid, confidence_threshold)){
            instance.reset();
        }
        return instance;
    }
};