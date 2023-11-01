#ifndef PREPROCESS_KERNEL_CUH
#define PREPROCESS_KERNEL_CUH

#include "cuda_tools.hpp"

namespace CUDAKernel{

    enum class NormType : int{
        None      = 0,
        MeanStd   = 1,
        AlphaBeta = 2
    };

    enum class ChannelType : int{
        None          = 0,
        Invert        = 1
    };

    struct Norm{
        float mean[3];
        float std[3];
        float alpha, beta;
        NormType type = NormType::None;
        ChannelType channel_type = ChannelType::None;

        // out = (x * alpha - mean) / std
        static Norm mean_std(const float mean[3], const float std[3], float alpha = 1/255.0f, ChannelType channel_type=ChannelType::None);

        // out = x * alpha + beta
        static Norm alpha_beta(float alpha, float beta = 0, ChannelType channel_type=ChannelType::None);

        // None
        static Norm None();
    };

    void resize_bilinear_and_normalize(
		uint8_t* src, int src_line_size, int src_width, int src_height, float* dst, int dst_width, int dst_height,
		const Norm& norm,
		cudaStream_t stream);

    void warp_affine_bilinear_and_normalize_plane(
        uint8_t* src, int src_line_size, int src_width, int src_height, 
        float* dst  , int dst_width, int dst_height,
        float* matrix_2_3, uint8_t const_value, const Norm& norm,
        cudaStream_t stream);

    void warp_affine_bilinear_and_normalize_focus(
        uint8_t* src, int src_line_size, int src_width, int src_height, 
        float* dst  , int dst_width, int dst_height,
        float* matrix_2_3, uint8_t const_value, const Norm& norm,
        cudaStream_t stream);

    void threshold_feature(
        float* feature_array, uint8_t* feature_output, int num_feature, int feature_length,int feature_width, float confidence,
        cudaStream_t stream
    );
    void threshold_feature_mat(
        float* feature_array, uint8_t* feature_output, int num_feature, int feature_length,int feature_width, float confidence,
        cudaStream_t stream
    );

    void resize_and_norm_plane(
            uint8_t* src, int src_width, int src_height, float* dst, int dst_width, int dst_height,int channels, Norm norm, 
            cudaStream_t stream);

    void convert_nv12_to_bgr_invoke(
        const uint8_t* y, const uint8_t* uv, int width, int height, 
        int linesize, uint8_t* dst, 
        cudaStream_t stream);
};

#endif // PREPROCESS_KERNEL_CUH