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


    void decode_kernel_invoker(
        float* predict, int num_bboxes, int num_classes, float confidence_threshold, 
        float* invert_affine_matrix, float* parray,
        int max_objects, cudaStream_t stream
    );

    void nms_kernel_invoker(
        float* parray, float nms_threshold, int max_objects, cudaStream_t stream
    );

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
        ResultArray,              // output
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
            const int NUM_BOX_ELEMENT = 791;    // left, top, right, bottom, confidence, keepflag(1keep,0ignore), top_feat(784)
            TRT::Tensor affin_matrix_device(TRT::DataType::Float);
            TRT::Tensor output_array_device(TRT::DataType::Float);
            TRT::Tensor output_bases_device(TRT::DataType::Float);
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

            vector<Job> fetch_jobs;
            while(get_jobs_and_wait(fetch_jobs, max_batch_size)){

                int infer_batch_size = fetch_jobs.size();
                input->resize_single_dim(0, infer_batch_size);
                input->resize_single_dim(2, input_height_);
                input->resize_single_dim(3, input_width_);
                bases_out->resize_single_dim(2, int(input_height_/4));
                bases_out->resize_single_dim(3, int(input_width_/4));
                output->resize_single_dim(1,calculate(input_height_, input_width_));

                // printf("*****%d-%d-%d-%d-%d\n", input->size(2), input->size(3),bases_out->size(2), bases_out->size(3),output->size(1));

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
                    checkCudaRuntime(cudaMemsetAsync(output_array_ptr, 0, sizeof(int), stream_));
                    // output->save_to_file("./inf/orig/used/output");

                    decode_kernel_invoker(image_pred_output, output->size(1), num_classes, confidence_threshold_, affine_matrix, output_array_ptr, MAX_IMAGE_BBOX, stream_);
                    // output_array_device.save_to_file("./inf/orig/used/output_device");
                    if(nms_method_ == NMSMethod::FastGPU){
                        nms_kernel_invoker(output_array_ptr, nms_threshold_, MAX_IMAGE_BBOX, stream_);
                    }
                    // output_array_device.save_to_file("./inf/orig/used/output_device");
                }
                output_array_device.to_cpu();
                Base base_out;
                const int kDataSize =  bases_out->size(1)*bases_out->size(2)*bases_out->size(3);
                base_out.base = shared_ptr<float>(new float[kDataSize], std::default_delete<float[]>());
                for(int ibatch = 0; ibatch < infer_batch_size; ++ibatch){
                    float* parray = output_array_device.cpu<float>(ibatch);
                    auto bases_out       = engine->output(0);
                    float* bases_parray = bases_out->cpu<float>(ibatch);
                    int count     = min(MAX_IMAGE_BBOX, (int)*parray);
                    auto& job     = fetch_jobs[ibatch];
                    auto& image_based_boxes   = job.output;
                    memcpy(base_out.base.get(), bases_parray, kDataSize*sizeof(float));
                    image_based_boxes.BasesArray.emplace_back(base_out);
                    
                    for(int i = 0; i < count; ++i){
                        float* pbox  = parray + 1 + i * NUM_BOX_ELEMENT;
                        int label    = pbox[5];
                        int keepflag = pbox[6];
                        if(keepflag == 1){
                            Obj obj;
                            obj.left       = pbox[0];
                            obj.top        = pbox[1];
                            obj.right      = pbox[2];
                            obj.bottom     = pbox[3];
                            obj.confidence = pbox[4];
                            obj.class_label = pbox[5];
                            memcpy(obj.top_feat, pbox + 7, sizeof(obj.top_feat)); 
                            image_based_boxes.BoxArray.emplace_back(obj);
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

            Size input_size(image.cols, image.rows);
            job.additional.compute(image.size(), input_size);
            
            tensor->set_stream(stream_);
            tensor->resize(1, channel, image.rows, image.cols);
            input_width_ = image.cols;
            input_height_ = image.rows;

            size_t size_image      = image.cols * image.rows * channel;
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
            memcpy(image_host, image.data, size_image);
            memcpy(affine_matrix_host, job.additional.d2i, sizeof(job.additional.d2i));
            checkCudaRuntime(cudaMemcpyAsync(image_device, image_host, size_image, cudaMemcpyHostToDevice, stream_));
            checkCudaRuntime(cudaMemcpyAsync(affine_matrix_device, affine_matrix_host, sizeof(job.additional.d2i), cudaMemcpyHostToDevice, stream_));

            CUDAKernel::warp_affine_bilinear_and_normalize_plane(
                image_device,         image.cols * channel,       image.cols,       image.rows, 
                tensor->gpu<float>(), image.cols,         image.rows, 
                affine_matrix_device, 0, 
                normalize_, stream_
            );

            // tensor->save_to_file("/media/ps/data/train/LQ/LQ/bdms/bdmask/workspace/imgs/process/1");

            return true;
        }

        virtual vector<shared_future<ResultArray>> commits(const vector<Mat>& images) override{
            return ControllerImpl::commits(images);
        }

        virtual std::shared_future<ResultArray> commit(const Mat& image) override{
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