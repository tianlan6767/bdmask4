#include "cfaAd.hpp"
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
   

namespace CfaAd{
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
    
    void morphologyOpening(uint8_t* mask_array_ptr, uint8_t* mask_output_ptr, int width, int height, int radius, cudaStream_t stream);

    void simple_normalize(float*data, uint8_t *output, float*max_result, float * min_result, int size, int min_max_result_size, float confidence_threshold,  cudaStream_t stream);

    void findMinMax(float * data, float *max_result, float * min_result, int size, cudaStream_t stream);

    void gaussian_filter(float * input, float * output, int width, int height, const float* kernel, cudaStream_t stream);

    void decode_kernel_invoker(
        float* predict, int height, int width, float confidence_threshold, uint8_t* parray, cudaStream_t stream
    );

    float gaussian_kernel(int x, int y) {
        return exp(-(x * x + y * y) / (2 * SIGMA * SIGMA));
    }

    void calculate_gaussian_kernal(float* kernel){
        for (int i = -KERNEL_SIZE / 2; i <= KERNEL_SIZE; ++i){
            for(int j = -KERNEL_SIZE / 2 ; j <= KERNEL_SIZE; ++j){
                int k_idx = (i+KERNEL_SIZE / 2) * KERNEL_SIZE + ( j+KERNEL_SIZE /2 );
                kernel[k_idx] = gaussian_kernel(i, j);
            }
        }
    }

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
        
        virtual bool startup(const string& file, ResultDetector::SpliceInfoArray &spliceInfoArray, int gpuid, float confidence_threshold, float mean[], float std[]){

            normalize_   = CUDAKernel::Norm::mean_std(mean, std, 1 / 255.0f);
            confidence_threshold_ = confidence_threshold;
            spliceInfoArray_ = spliceInfoArray;

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
            TRT::Tensor output_array_device(TRT::DataType::Float);
            TRT::Tensor mask_array_device(TRT::DataType::UInt8);   // 转换为最终大图的mask
            TRT::Tensor mask_array_tmp_device(TRT::DataType::UInt8); // 中间输出mask
            TRT::Tensor mask_array_opening_out_device(TRT::DataType::UInt8); // 中间开运算输出mask

            TRT::Tensor gaussian_kernel_device(TRT::DataType::Float);
            TRT::Tensor min_Max_device(TRT::DataType::Float);
            TRT::Tensor max_result_device(TRT::DataType::Float);
            TRT::Tensor min_result_device(TRT::DataType::Float);


            int max_batch_size = 1;
            auto input         = engine->input();
            auto output        = engine->output(0);
            output_height_     = output->size(2);
            output_width_      = output->size(3);
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
            min_Max_device.resize(2).to_gpu();

            // 这里的 1 + MAX_IMAGE_BBOX结构是，counter + bboxes ...
            output_array_device.resize(max_batch_size, output_height_, output_width_).to_gpu(); 
            mask_array_device.resize(max_batch_size, output_height_, output_width_).to_gpu(); 
            mask_array_tmp_device.resize(max_batch_size, output_height_, output_width_).to_gpu();
            mask_array_opening_out_device.resize(max_batch_size, output_height_, output_width_).to_gpu();
            gaussian_kernel_device.resize(KERNEL_SIZE, KERNEL_SIZE).to_gpu();
            int output_size = max_batch_size * output_width_ * output_height_;
            int min_max_result_size = output_size + 512 - 1 / 512;
            max_result_device.resize(1, min_max_result_size).to_gpu();
            min_result_device.resize(1, min_max_result_size).to_gpu();
            calculate_gaussian_kernal(gaussian_kernel_device.cpu<float>());

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
                    auto& job                  = fetch_jobs[ibatch];
                    float* image_pred_output   = output->gpu<float>(ibatch);
                    float* output_array_ptr    = output_array_device.gpu<float>(ibatch);
                    uint8_t* mask_array_ptr    = mask_array_device.gpu<uint8_t>(ibatch);
                    auto affine_matrix_i2d     = affine_matrix_device_i2d.gpu<float>(ibatch);
                    auto affine_matrix_cpu_i2d = affine_matrix_device_i2d.cpu<float>(ibatch);
                    auto max_result_ptr        = max_result_device.gpu<float>(ibatch);
                    auto min_result_ptr        = min_result_device.gpu<float>(ibatch);
                    auto kernel_ptr            =  gaussian_kernel_device.gpu<float>();
                    auto mask_array_tmp_ptr    = mask_array_tmp_device.gpu<uint8_t>(ibatch);
                    auto mask_array_opening_out_ptr    = mask_array_opening_out_device.gpu<uint8_t>(ibatch);
                    memcpy(affine_matrix_cpu_i2d, job.additional.i2d, sizeof(job.additional.i2d));

                    checkCudaRuntime(cudaMemcpyAsync(affine_matrix_i2d, affine_matrix_cpu_i2d, sizeof(job.additional.i2d), cudaMemcpyHostToDevice, stream_));
                    checkCudaRuntime(cudaMemsetAsync(output_array_ptr, 0, sizeof(float)*(output_height_ * output_width_), stream_));
                    checkCudaRuntime(cudaMemsetAsync(max_result_ptr, 0, sizeof(float)*min_max_result_size, stream_));
                    checkCudaRuntime(cudaMemsetAsync(min_result_ptr, 0, sizeof(float)*min_max_result_size, stream_));
                    
                    checkCudaRuntime(cudaMemsetAsync(mask_array_ptr, 0, sizeof(uint8_t)*(orig_inp_height_ * orig_inp_width_), stream_));
                    checkCudaRuntime(cudaMemsetAsync(mask_array_tmp_ptr, 0, sizeof(uint8_t)*(output_height_ * output_height_), stream_));
                    checkCudaRuntime(cudaMemsetAsync(mask_array_opening_out_ptr, 0, sizeof(uint8_t)*(output_height_ * output_height_), stream_));
                    // output->save_to_file("/media/ps/data1/train/LQ/task/bdm/bdmask/workspace/models/cfa/output");
                    gaussian_filter(image_pred_output, output_array_ptr, output_width_, output_height_, kernel_ptr, stream_);
                    // output_array_device.save_to_file("/media/ps/data1/train/LQ/task/bdm/bdmask/workspace/models/cfa/output_array_device_used");
                    findMinMax(output_array_ptr, max_result_ptr, min_result_ptr,  output_width_*output_height_, stream_);
                    // 进行归一化并过滤
                    simple_normalize(output_array_ptr, mask_array_tmp_ptr, max_result_ptr, min_result_ptr, output_width_*output_height_, min_max_result_size, confidence_threshold_, stream_);
                    
                    // 进行开运算
                    morphologyOpening(mask_array_tmp_ptr, mask_array_opening_out_ptr, output_width_, output_height_, 1, stream_);
                    
                    // 还原大图
                    CUDAKernel::warp_affine_bilinear_mask(
                        mask_array_opening_out_ptr,         output_width_ * 1,       output_width_,       output_height_, 
                        mask_array_ptr,        orig_inp_height_,         orig_inp_width_, 
                        affine_matrix_i2d, 0, stream_);
                }

                output_array_device.to_cpu();
                mask_array_device.to_cpu();
                for(int ibatch = 0; ibatch < infer_batch_size; ++ibatch){
                    auto& job     = fetch_jobs[ibatch];
                    uint8_t* mask_array_host  = mask_array_device.cpu<uint8_t>(ibatch);
                    auto& image_based_boxes   = job.output;

                    cv::Mat output_mask(orig_inp_height_, orig_inp_width_, CV_8UC1, mask_array_host);
                    // cv::imwrite("/media/ps/data1/train/LQ/task/bdm/bdmask/workspace/models/cfa/inf/out.jpg", output_mask);
                    for (int idx=0; idx < spliceInfoArray_.size(); ++idx){
                        auto imn = spliceInfoArray_[idx].image_name;
                        auto crop_h = spliceInfoArray_[idx].crop_h;
                        auto crop_w = spliceInfoArray_[idx].crop_w;
                        auto splice_x = spliceInfoArray_[idx].splice_x;
                        auto splice_y = spliceInfoArray_[idx].splice_y;
                        auto region_idx = spliceInfoArray_[idx].region_idx;
                        // 创建感兴趣区域的矩形
                        cv::Rect roi(splice_x, splice_y, crop_w, crop_h);

                        // 从 mask_array_host 数组中截取感兴趣区域
                        cv::Mat region = output_mask(roi);
                        vector<cv::Point> non_zeros_pts_idx;
                        cv::findNonZero(region, non_zeros_pts_idx);
                        // 如果当前区域没有缺陷的话, 跳过。
                        if(non_zeros_pts_idx.size() == 0){
                            continue;
                        }
                        std::vector<std::vector<cv::Point>> contours;
                        cv::findContours(region, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
                        int num = 0;
                        for (const auto & contour : contours){
                            if(contour.size() <= 3) continue;
                            cv::Rect bbox = cv::boundingRect(contour);
                            cv::Rect roi_ng(bbox.x, bbox.y, bbox.width, bbox.height);
                            cv::Mat region_ng = region(roi_ng);

                            if (cv::countNonZero(region_ng) < 500 || bbox.width <= 5 || bbox.height <=5) continue;
                            // 计算均值和最大值
                            cv::Point minLoc, maxLoc;
                            double minVal, maxVal;
                            auto meanVal = cv::mean(region_ng, region_ng!=0);
                            cv::minMaxLoc(region_ng, &minVal, &maxVal, &minLoc, &maxLoc);
                            if ((meanVal[0]/ 255) < 0.09) continue;
                            // 输出结果
                            Box result_object_box(float(bbox.x+splice_x), float(bbox.y+splice_y), float(bbox.x+splice_x+bbox.width),float(bbox.y+splice_y+bbox.height), 0.0f, region_idx, meanVal.val[0] / 255, float(maxVal) / 255);
                            result_object_box.seg = make_shared<InstanceSegmentMap>(bbox.width, bbox.height);
                            uint8_t* ng_mask_out_host = result_object_box.seg->data;
                            // 创建 `region_ng` 的副本
                            cv::Mat region_ng_cp = region_ng.clone();
                            memcpy(ng_mask_out_host, region_ng_cp.data, bbox.width * bbox.height);
                            image_based_boxes.emplace_back(result_object_box);
                            num++;
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
        float kernel_[KERNEL_SIZE * KERNEL_SIZE];
        NMSMethod nms_method_       = NMSMethod::FastGPU;
        TRT::CUStream stream_       = nullptr;
        ResultDetector::SpliceInfoArray spliceInfoArray_;
        CUDAKernel::Norm normalize_;
    };


    shared_ptr<Infer> create_infer(const string& engine_file, SpliceInfoArray &spliceInfoArray, int gpuid, float confidence_threshold, float mean[3], float std[3]){
        shared_ptr<InferImpl> instance(new InferImpl());
        if(!instance->startup(engine_file, spliceInfoArray, gpuid, confidence_threshold,  mean, std)){
            instance.reset();
        }
        return instance;
    }
};