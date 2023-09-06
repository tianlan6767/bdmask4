#include "blender.hpp"
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

namespace Blender{
    using namespace cv;
    using namespace std;
    
    using ControllerImpl = InferController
    <
        commit_input,           // input
        feature,                // output
        tuple<string, int>     // start param
    >;


    class MatPool {
    public:
    MatPool(int pool_size, int height, int width) :
        pool_size_(pool_size), height_(height), width_(width) {
        for (int i = 0; i < pool_size_; ++i) {
        Mat mat = Mat::zeros(height_, width_, CV_8UC1);
        pool_.push_back(mat);
        }
    }

    // 获取 Mat 对象
    Mat get_mat() {
        if (pool_.empty()) {
        return Mat::zeros(height_, width_, CV_8UC1);
        } else {
        Mat mat = pool_.front();
        pool_.pop_front();
        return mat;
        }
    }

    // 归还 Mat 对象
    void return_mat(Mat mat) {
        if (pool_.size() < pool_size_) {
        pool_.push_back(mat);
        }
    }

    private:
    int pool_size_;
    int height_;
    int width_;
    std::deque<Mat> pool_;
    };
    class InferImpl : public Infer, public ControllerImpl{
    public:
        /** 要求在InferImpl里面执行stop，而不是在基类执行stop **/
        virtual ~InferImpl(){
            TRT::set_device(gpu_);
            stop();
        }
        
        virtual bool startup(const string& file, int gpuid){

            float mean[] = {0.5f, 0.5f, 0.5f};
            float std[]  = {0.5f, 0.5f, 0.5f};
            normalize_   = CUDAKernel::Norm::mean_std(mean, std, 1.0f / 255.0f, CUDAKernel::ChannelType::Invert);
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
            TRT::Tensor feature_output_device(TRT::DataType::UInt8);
            int max_batch_size = engine->get_max_batch_size();
            auto base_input         = engine->input(0);
            auto box_input          = engine->input(1);
            auto top_feat_input     = engine->input(2);
            auto output        = engine->output();

            input_width_       = base_input->size(3);
            input_height_      = base_input->size(2);
            feature_width_     = output->size(2);
            feature_height_    = output->size(1);
            num_batch_size_    = box_input->size(0);
            box_element_       = box_input->size(1);
            top_feat_element_  = top_feat_input->size(1);
            tensor_allocator_  = make_shared<MonopolyAllocator<TRT::Tensor>>(num_batch_size_ * 2);
            stream_            = engine->get_stream();
            gpu_               = gpuid;
            result.set_value(true);
            
            base_input->resize_single_dim(0, max_batch_size).to_gpu();
            output->resize_single_dim(0, num_batch_size_).to_gpu();
            feature_output_device.resize(num_batch_size_, num_batch_size_*feature_width_*feature_height_).to_gpu();

            vector<Job> fetch_jobs;
            // MatPool mat_pool(4, feature_height_, feature_width_);
            while(get_jobs_and_wait(fetch_jobs, max_batch_size)){
                int infer_batch_size = fetch_jobs.size();
                base_input->resize_single_dim(0, infer_batch_size);
                for(int ibatch = 0; ibatch < infer_batch_size; ++ibatch){
                    auto& job  = fetch_jobs[ibatch];
                    auto& mono = job.mono_tensor->data();
                    box_input->copy_from_gpu(base_input->offset(ibatch), mono->get_workspace()->gpu(), num_batch_size_* box_element_);
                    top_feat_input->copy_from_gpu(base_input->offset(ibatch), mono->get_workspace()->gpu()+ iLogger::upbound(num_batch_size_* box_element_ * sizeof(float), 32), num_batch_size_ * top_feat_element_);
                    base_input->copy_from_gpu(base_input->offset(ibatch), mono->get_workspace()->gpu()+ iLogger::upbound(num_batch_size_* box_element_ * sizeof(float), 32) + iLogger::upbound(num_batch_size_* top_feat_element_ * sizeof(float), 32), input_width_ * input_height_*4);
                    job.mono_tensor->release();
                }
                // auto st = iLogger::timestamp_now_float();
                engine->forward(false);
                feature_output_device.to_gpu(false);
                // box_input->save_to_file("./box_input");
                // top_feat_input->save_to_file("./top_feat_input");
                // base_input->save_to_file("./base_input");

                
                
                // auto et1 = iLogger::timestamp_now_float();
                for(int ibatch = 0; ibatch < infer_batch_size; ++ibatch){
                    auto& job                 = fetch_jobs[ibatch];
                    float* image_output = output->gpu<float>(ibatch);
                    uint8_t* feature_output_ptr = feature_output_device.gpu<uint8_t>(ibatch);
                    checkCudaRuntime(cudaMemsetAsync(feature_output_ptr, 0, sizeof(int), stream_));

                    // auto et2_0 = iLogger::timestamp_now_float();
                    feature feature_mat = Mat::zeros(feature_height_, feature_width_, CV_8UC1);

                    // CUDAKernel::threshold_feature(image_output, feature_output_ptr, num_batch_size_, output->size(1), output->size(2), 0.5f, stream_);
                    // auto et2_1 = iLogger::timestamp_now_float();

                    CUDAKernel::threshold_feature_mat(image_output, feature_output_ptr, num_batch_size_, output->size(1), output->size(2), 0.5f, stream_);
                    cudaMemcpy(feature_mat.data, feature_output_ptr, sizeof(uint8_t) * feature_height_* feature_width_, cudaMemcpyDeviceToHost);
                    job.pro->set_value(feature_mat);
                    // mat_pool.return_mat(feature_mat);
                    // auto et2_2 = iLogger::timestamp_now_float();
                    // printf("*******当前拷贝数据推理耗时:******%f******%f******\n", et2_2 - et2_1, et2_1 - et2_0);

                }
                // auto et2 = iLogger::timestamp_now_float();
                // printf("*******当前只有mask推理耗时:**********%f*******%f*************\n", et2 - et1, et1 - st);
                fetch_jobs.clear();
            }
            
            stream_ = nullptr;
            tensor_allocator_.reset();
            INFO("Engine destroy.");
        }

        virtual bool preprocess(Job& job, const commit_input& input) override{
            
            auto& image_feature = get<0>(input);
            auto& box = get<1>(input);
            auto& top_feat = get<2>(input);
            // cout << "当前box:" << box.get()[0] <<"**"<< box.get()[1] <<"**"<< box.get()[2] <<"**"<<box.get()[3]<<"**"<<box.get()[4]<< endl;

            // cout << "当前feat:" << top_feat[0] << endl;


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

            
            tensor->set_stream(stream_);

            size_t size_image_feature_ =   iLogger::upbound(input_height_*input_width_ * 4 * sizeof(float), 32);

            size_t size_box_num     = num_batch_size_ * box_element_;
            size_t size_box_         =     iLogger::upbound(size_box_num*sizeof(float), 32);

            size_t size_top_feat_num     = num_batch_size_ * top_feat_element_;
            size_t size_top_feat_     =  iLogger::upbound(size_top_feat_num * sizeof(float), 32);
            
            size_t all_element = size_image_feature_ + size_box_ + size_top_feat_;

            auto workspace         = tensor->get_workspace();
            
            uint8_t* gpu_workspace      = (uint8_t*)workspace->gpu(size_image_feature_ + size_box_ + size_top_feat_);
            float* box_device         =   (float*)gpu_workspace;
            float* top_feat_device    =   (float*)(size_box_ + gpu_workspace);
            float* image_feature_device = (float*)(size_top_feat_ + size_box_ + gpu_workspace);
            

            uint8_t* cpu_workspace       = (uint8_t*)workspace->cpu(size_image_feature_ + size_box_ + size_top_feat_);
            float* box_host            =   (float*)cpu_workspace;
            float* top_feat_host       =   (float*)(size_box_ + cpu_workspace);
            float* image_feature_host    = (float*)(size_top_feat_ + size_box_ + cpu_workspace);

            // speed up
            memcpy(image_feature_host, image_feature.get(), size_image_feature_);
            memcpy(box_host, box.get(), size_box_);
            memcpy(top_feat_host, top_feat, size_top_feat_);

            checkCudaRuntime(cudaMemcpyAsync(image_feature_device, image_feature_host, size_image_feature_, cudaMemcpyHostToDevice, stream_));
            checkCudaRuntime(cudaMemcpyAsync(box_device,           box_host,           size_box_,           cudaMemcpyHostToDevice, stream_));
            checkCudaRuntime(cudaMemcpyAsync(top_feat_device,      top_feat_host,      size_top_feat_,      cudaMemcpyHostToDevice, stream_));
            tensor->gpu<float>();
            return true;
        }

        virtual vector<shared_future<feature>> commits(const vector<commit_input>& inputs) override{
            return ControllerImpl::commits(inputs);
        }

        virtual std::shared_future<feature> commit(const commit_input& input) override{
            return ControllerImpl::commit(input);
        }

    private:
        int input_width_            = 0;
        int input_height_           = 0;
        int feature_width_          = 0;
        int feature_height_         = 0;
        int box_element_            = 0;
        int num_batch_size_         = 0;
        int top_feat_element_       = 0;
        int gpu_                    = 0;
        float confindence_          = 0.5f;

        TRT::CUStream stream_       = nullptr;
        CUDAKernel::Norm normalize_;
    };


    shared_ptr<Infer> create_infer(const string& engine_file, int gpuid){
        shared_ptr<InferImpl> instance(new InferImpl());
        if(!instance->startup(engine_file, gpuid)){
            instance.reset();
        }
        return instance;
    }
};