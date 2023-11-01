#include "fcos.hpp"
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

namespace Fcos{
    using namespace cv;
    using namespace std;


    InstanceSegmentMap::InstanceSegmentMap(int width, int height) {
        this->width = width;
        this->height = height;
        checkCudaRuntime(cudaMallocHost(&this->data, width * height));
    }

    InstanceSegmentMap::~InstanceSegmentMap() {
        if (this->data) {
            checkCudaRuntime(cudaFreeHost(this->data));
            this->data = nullptr;
        }
        this->width = 0;
        this->height = 0;
    }

    // struct AffineMatrix{
    //     float i2d[6];       // image to dst(network), 2x3 matrix
    //     float d2i[6];       // dst to image, 2x3 matrix

    //     void compute(const cv::Size& from, const cv::Size& to){
    //         float scale_x = to.width / (float)from.width;
    //         float scale_y = to.height / (float)from.height;
    //         float scale = std::min(scale_x, scale_y);

    //         i2d[0] = scale;  i2d[1] = 0;  i2d[2] = -scale * from.width  * 0.5  + to.width * 0.5 + scale * 0.5 - 0.5;
    //         i2d[3] = 0;  i2d[4] = scale;  i2d[5] = -scale * from.height * 0.5 + to.height * 0.5 + scale * 0.5 - 0.5;

    //         // 有了i2d矩阵，我们求其逆矩阵，即可得到d2i（用以解码时还原到原始图像分辨率上）
    //         cv::Mat m2x3_i2d(2, 3, CV_32F, i2d);
    //         cv::Mat m2x3_d2i(2, 3, CV_32F, d2i);
    //         cv::invertAffineTransform(m2x3_i2d, m2x3_d2i);
    //     }

    //     cv::Mat i2d_mat(){
    //         return cv::Mat(2, 3, CV_32F, i2d);
    //     }
    // };

    struct AffineMatrix{
        float i2d[6];       // image to dst(network), 2x3 matrix
        float d2i[6];       // dst to image, 2x3 matrix

        void compute(const cv::Size& from, const cv::Size& to){
            assert(from.width > 0 && from.height > 0 && to.width > 0 && to.height > 0 && to.width >= from.width && to.height >= from.height);
            i2d[0] = 1.0f;  i2d[1] = 0;  i2d[2] = 0;
            i2d[3] = 0;  i2d[4] = 1.0f;  i2d[5] = 0;

            // 有了i2d矩阵，我们求其逆矩阵，即可得到d2i（用以解码时还原到原始图像分辨率上）
            cv::Mat m2x3_i2d(2, 3, CV_32F, i2d);
            cv::Mat m2x3_d2i(2, 3, CV_32F, d2i);
            cv::invertAffineTransform(m2x3_i2d, m2x3_d2i);
        }

        cv::Mat i2d_mat(){
            return cv::Mat(2, 3, CV_32F, i2d);
        }
    };


    void generate_grid(float* box, int N, int height, int width, float* box_grid, cudaStream_t stream);
    
    void decode_roialign(float* bottom_data,float spatial_scale, int channels, int height, int width, int pooled_height, int pooled_width, float sampling_ratio, float* bottom_rois,
        float* top_data, bool aligned, cudaStream_t stream);

    void decode_interpolate(float* src, int channels, int src_height, int src_width, float* dst, int dst_height, int dst_width,float scale_factor,
            cudaStream_t stream);
    void decode_softmax(float* input, int height, int width, cudaStream_t stream);

    void decode_mul_sum_sigmod(float * top_data, float * feat_out_tensor, int height, int width,  float*mask_pred, cudaStream_t stream);

    void decode_grid_sample(float* mask, int mask_height, int mask_width, float* box_grid,  int grid_height, int grid_width, uint8_t* box_mask, cudaStream_t stream);

    void decode_kernel_invoker(
        float* predict, int num_bboxes, int num_classes, float confidence_threshold, 
        float* invert_affine_matrix, float* parray,
        int max_objects, cudaStream_t stream
    );

    void nms_kernel_invoker(
        float* parray, float nms_threshold, int max_objects, cudaStream_t stream
    );

    void recount_box(float* parray, int h, int w, int max_objects, cudaStream_t stream);

    int calculate(int h, int w){
        int array[] = {8, 16, 32, 64, 128};
        int result = 0;
        auto feature_num = [&](int value) {
            
            return static_cast<int>(std::ceil(static_cast<double>(h) / value) * std::ceil(static_cast<double>(w) / value));
        };
        for(int i=0;i < sizeof(array)/ sizeof(array[0]); i++){
            result += feature_num(array[i]);
        }
        return result;
    }



    using ControllerImpl = InferController
    <
        Mat,                    // input
        BoxArray,              // output
        tuple<string, int>,     // start param
        AffineMatrix            // additional
    >;
    class InferImpl : public Infer, public ControllerImpl{
    public:
        /** 要求在InferImpl里面执行stop，而不是在基类执行stop **/
        virtual ~InferImpl(){
            stop();
        }
        
        virtual bool startup(const string& file, int gpuid, float confidence_threshold, float mean[], float std[], float nms_threshold, NMSMethod nms_method){

            normalize_   = CUDAKernel::Norm::mean_std(mean, std, 1.0f);
            confidence_threshold_ = confidence_threshold;
            nms_threshold_        = nms_threshold;
            nms_method_           = nms_method;
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

            const int MAX_IMAGE_BBOX = 1024;
            const int NUM_BOX_ELEMENT = 792;    // left, top, right, bottom, confidence, keepflag(1keep,0ignore), num_index, top_feat(784)
            const int FEAT_DIM        = 14;
            const int MASK_DIM        = 56;
            TRT::Tensor affin_matrix_device(TRT::DataType::Float);
            TRT::Tensor output_array_device(TRT::DataType::Float);
            TRT::Tensor output_bases_device(TRT::DataType::Float);
            TRT::Tensor box_output_device(TRT::DataType::Float);
            
            TRT::Tensor box_device(TRT::DataType::Float);
            TRT::Tensor feat_device(TRT::DataType::Float);
            TRT::Tensor feat_out_device(TRT::DataType::Float);
            TRT::Tensor top_device(TRT::DataType::Float);
            TRT::Tensor mask_pred_device(TRT::DataType::Float);
            TRT::Tensor box_grid_device(TRT::DataType::Float);
            TRT::Tensor box_mask_device(TRT::DataType::UInt8);

            int max_batch_size = engine->get_max_batch_size();
            auto input         = engine->input();
            auto output        = engine->output(1);
            auto bases_out       = engine->output(0);
            int num_classes      = output->size(2) - 789;
            
            input_width_       = input->size(3);
            input_height_      = input->size(2);
            tensor_allocator_  = make_shared<MonopolyAllocator<TRT::Tensor>>(max_batch_size * 2);
            stream_            = engine->get_stream();
            gpu_               = gpuid;
            result.set_value(true);

            input->resize_single_dim(0, max_batch_size).to_gpu();
            affin_matrix_device.set_stream(stream_);

            // 这里8个值的目的是保证 8 * sizeof(float) % 32 == 0
            affin_matrix_device.resize(max_batch_size, 8).to_gpu();

            // 这里的 1 + MAX_IMAGE_BBOX结构是，counter + bboxes ...
            output_array_device.resize(max_batch_size, 1 + MAX_IMAGE_BBOX * NUM_BOX_ELEMENT).to_gpu(); 
            output_bases_device.resize(max_batch_size, bases_out->size(1)*bases_out->size(2)*bases_out->size(3)).to_gpu();

            // mask参数
            top_device.resize(max_batch_size, bases_out->size(1), MASK_DIM, MASK_DIM).to_gpu();
            feat_out_device.resize(max_batch_size, bases_out->size(1), MASK_DIM, MASK_DIM).to_gpu();
            mask_pred_device.resize(MASK_DIM, MASK_DIM).to_gpu();

            box_device.resize(max_batch_size, 5); // num_batch, left, top, right, bottom
            feat_device.resize(max_batch_size, bases_out->size(1), FEAT_DIM, FEAT_DIM).to_gpu();

            vector<Job> fetch_jobs;
            while(get_jobs_and_wait(fetch_jobs, max_batch_size)){

                int infer_batch_size = fetch_jobs.size();
                input->resize_single_dim(0, infer_batch_size);
                input->resize_single_dim(2, input_height_);
                input->resize_single_dim(3, input_width_);
                bases_out->resize_single_dim(2, int(input_height_/4));
                bases_out->resize_single_dim(3, int(input_width_/4));
                output->resize_single_dim(1,calculate(input_height_, input_width_));

                for(int ibatch = 0; ibatch < infer_batch_size; ++ibatch){
                    auto& job  = fetch_jobs[ibatch];
                    auto& mono = job.mono_tensor->data();
                    affin_matrix_device.copy_from_gpu(affin_matrix_device.offset(ibatch), mono->get_workspace()->gpu(), 6);
                    input->copy_from_gpu(input->offset(ibatch), mono->gpu(), mono->count());
                    job.mono_tensor->release();
                }
                engine->forward(false);
                output_array_device.to_gpu(false);
                for(int ibatch = 0; ibatch < infer_batch_size; ++ibatch){
                    auto& job                 = fetch_jobs[ibatch];
                    float* image_pred_output = output->gpu<float>(ibatch);
                    
                    float* output_array_ptr   = output_array_device.gpu<float>(ibatch);
                    auto affine_matrix        = affin_matrix_device.gpu<float>(ibatch);
                    checkCudaRuntime(cudaMemsetAsync(output_array_ptr, 0, sizeof(float)*(1 + MAX_IMAGE_BBOX * NUM_BOX_ELEMENT)*(max_batch_size), stream_));

                    decode_kernel_invoker(image_pred_output, output->size(1), num_classes, confidence_threshold_, affine_matrix, output_array_ptr, MAX_IMAGE_BBOX, stream_);
                    recount_box(output_array_ptr, input_height_, input_width_, MAX_IMAGE_BBOX, stream_);
                    if(nms_method_ == NMSMethod::FastGPU){
                        nms_kernel_invoker(output_array_ptr, nms_threshold_, MAX_IMAGE_BBOX, stream_);
                    }
                }

                // compute mask
                output_array_device.to_cpu();
                for(int ibatch = 0; ibatch < infer_batch_size; ++ibatch){
                    float* parray = output_array_device.cpu<float>(ibatch);
                    int count     = min(MAX_IMAGE_BBOX, (int)*parray);
                    
                    auto bases_out       = engine->output(0);
                    float* base_tensor = bases_out->gpu<float>(ibatch);

                    auto& job     = fetch_jobs[ibatch];
                    auto& image_based_boxes   = job.output;
                    for(int i = 0; i < count; ++i){
                        float* pbox  = parray + 1 + i * NUM_BOX_ELEMENT;
                        int keepflag = pbox[6];
                        if(keepflag == 1){
                            pbox[0] = max(0.f, min(float(input_width_), pbox[0]));
                            pbox[2] = max(0.f, min(float(input_width_), pbox[2]));
                            pbox[1] = max(0.f, min(float(input_height_), pbox[1]));
                            pbox[3] = max(0.f, min(float(input_height_), pbox[3]));
                            Box result_object_box(pbox[0], pbox[1], pbox[2], pbox[3], pbox[4], pbox[5]);
                            int box_mask_height = pbox[3] - pbox[1] + 0.5f;
                            int box_mask_width  = pbox[2] - pbox[0] + 0.5f;
                            box_grid_device.resize(1, box_mask_height, box_mask_width, 2);
                            box_mask_device.resize(box_mask_height, box_mask_width);
                            float* box_tensor = box_device.gpu<float>();
                            float* feat_tensor = feat_device.gpu<float>();
                            float* box_grid = box_grid_device.gpu<float>();
                            float* top_data = top_device.gpu<float>();
                            float* feat_out_tensor = feat_out_device.gpu<float>();
                            float* mask_pred       = mask_pred_device.gpu<float>();
                            uint8_t* box_mask = box_mask_device.gpu<uint8_t>();
                            result_object_box.seg =
                                make_shared<InstanceSegmentMap>(box_mask_width, box_mask_height);
                            uint8_t* mask_out_host = result_object_box.seg->data;
                            checkCudaRuntime(cudaMemcpyAsync(box_tensor+1, pbox, 4 * sizeof(float), cudaMemcpyHostToDevice, stream_));
                            checkCudaRuntime(cudaMemcpyAsync(feat_tensor, pbox + 7, 784 * sizeof(float), cudaMemcpyHostToDevice, stream_));
                            float* box_tensor_cpu = box_device.cpu<float>();

                            
                            // 调用 CUDA 核函数  
                            generate_grid(box_tensor, 1, box_mask_height, box_mask_width, box_grid, stream_);
                            decode_roialign(base_tensor, 0.25f, bases_out->size(1), bases_out->size(2), bases_out->size(3), MASK_DIM, MASK_DIM, 1, box_tensor, top_data, true, stream_);
                            decode_interpolate(feat_tensor, bases_out->size(1), FEAT_DIM, FEAT_DIM, feat_out_tensor, MASK_DIM, MASK_DIM, 0.25f, stream_);
                            decode_softmax(feat_out_tensor, MASK_DIM, MASK_DIM, stream_);
                            decode_mul_sum_sigmod(top_data, feat_out_tensor, MASK_DIM, MASK_DIM,  mask_pred, stream_);
                            decode_grid_sample(mask_pred, MASK_DIM, MASK_DIM, box_grid,  box_mask_height, box_mask_width, box_mask, stream_);
                            checkCudaRuntime(cudaMemcpyAsync(mask_out_host, box_mask,
                                           box_mask_height * box_mask_width * sizeof(uint8_t),
                                           cudaMemcpyDeviceToHost, stream_));
                            cudaStreamSynchronize(stream_);
                            image_based_boxes.emplace_back(result_object_box);
                        }
                    }
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

            tensor->set_stream(stream_);
            input_width_ = iLogger::upbound(image.cols, 32);
            input_height_ = iLogger::upbound(image.rows, 32);
            tensor->resize(1, channel, input_height_, input_width_);
            cv::Size image_size = image.size();
            Size input_size(input_width_, input_height_);
            job.additional.compute(image_size, input_size);

            size_t size_image      = input_width_ * input_height_ * channel;
            size_t size_image_old  = image_size.width * image_size.height * channel;
            size_t size_matrix     = iLogger::upbound(sizeof(job.additional.d2i), 32);
            auto workspace         = tensor->get_workspace();
            uint8_t* gpu_workspace        = (uint8_t*)workspace->gpu(size_matrix + size_image);
            float*   affine_matrix_device = (float*)gpu_workspace;
            uint8_t* image_device         = size_matrix + gpu_workspace;

            uint8_t* cpu_workspace        = (uint8_t*)workspace->cpu(size_matrix + size_image);
            float* affine_matrix_host     = (float*)cpu_workspace;
            uint8_t* image_host           = size_matrix + cpu_workspace;

            //checkCudaRuntime(cudaMemcpyAsync(image_host,   image.data, size_image, cudaMemcpyHostToHost,   stream_));
            // speed up
            // memcpy(image_host, image.data, size_image);
            // TRT::Tensor src_device(TRT::DataType::UInt8);
            // src_device.resize(image.rows, image.cols, channel);
            // uint8_t* image_device = src_device.gpu<uint8_t>();
            memcpy(image_host, image.data, size_image_old);
            memcpy(affine_matrix_host, job.additional.d2i, sizeof(job.additional.d2i));
            checkCudaRuntime(cudaMemcpyAsync(image_device, image_host, size_image, cudaMemcpyHostToDevice, stream_));
            checkCudaRuntime(cudaMemcpyAsync(image_device, image_host, size_image, cudaMemcpyHostToDevice, stream_));
            checkCudaRuntime(cudaMemcpyAsync(affine_matrix_device, affine_matrix_host, sizeof(job.additional.d2i), cudaMemcpyHostToDevice, stream_));

            CUDAKernel::warp_affine_bilinear_and_normalize_plane(
                image_device,         image.cols * channel,       image.cols,       image.rows, 
                tensor->gpu<float>(), input_width_,         input_height_, 
                affine_matrix_device, 0, 
                normalize_, stream_
            );

            // 将图片粘贴在左上角
            // CUDAKernel::resize_and_norm_plane(
            //     image_device, image.cols, image.rows, tensor->gpu<float>(), input_width_, input_height_, channel, normalize_, stream_   
            // );

            // tensor->save_to_file("/media/ps/data/train/LQ/task/bdm/bdmask/workspace/models/JT/JT-imgs/2222/ttt");

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
        int gpu_                    = 0;
        float confidence_threshold_ = 0;
        float nms_threshold_        = 0;
        NMSMethod nms_method_       = NMSMethod::FastGPU;
        TRT::CUStream stream_       = nullptr;
        CUDAKernel::Norm normalize_;
    };


    shared_ptr<Infer> create_infer(const string& engine_file, int gpuid, float confidence_threshold,  float mean[], float std[], float nms_threshold, NMSMethod nms_method){
        shared_ptr<InferImpl> instance(new InferImpl());
        if(!instance->startup(engine_file, gpuid, confidence_threshold, mean, std,  nms_threshold, nms_method)){
            instance.reset();
        }
        return instance;
    }
};