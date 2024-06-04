#include "NvInfer.h"
#include "NvInferPlugin.h"
#include <cuda_runtime_api.h>
#include <iostream>

using namespace nvinfer1;

class HSwishPlugin : public IPluginV2DynamicExt {
public:
    HSwishPlugin() {}

    HSwishPlugin(const void* data, size_t length) {}

    int getNbOutputs() const override {
        return 1;
    }

    DimsExprs getOutputDimensions(int outputIndex, const DimsExprs* inputs, int nbInputs, IExprBuilder& exprBuilder) override {
        return inputs[0];
    }

    int initialize() override {
        return 0;
    }

    void terminate() override {}

    size_t getWorkspaceSize(const PluginTensorDesc* inputs, int nbInputs, const PluginTensorDesc* outputs, int nbOutputs) const override {
        return 0;
    }

    int enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) override {
        // Launch CUDA kernel for h-swish
        // For simplicity, directly implementing the CPU version here. Replace with CUDA kernel for actual use.
        const float* input = static_cast<const float*>(inputs[0]);
        float* output = static_cast<float*>(outputs[0]);
        int volume = 1;
        for (int i = 0; i < inputDesc[0].dims.nbDims; ++i) {
            volume *= inputDesc[0].dims.d[i];
        }
        for (int i = 0; i < volume; ++i) {
            float x = input[i];
            output[i] = x * std::min(std::max(x + 3.0f, 0.0f), 6.0f) / 6.0f;
        }
        return 0;
    }

    size_t getSerializationSize() const override {
        return 0;
    }

    void serialize(void* buffer) const override {}

    void destroy() override {
        delete this;
    }

    IPluginV2DynamicExt* clone() const override {
        return new HSwishPlugin();
    }

    void setPluginNamespace(const char* libNamespace) override {}

    const char* getPluginNamespace() const override {
        return "";
    }

    DataType getOutputDataType(int index, const DataType* inputTypes, int nbInputs) const override {
        return inputTypes[0];
    }

    bool supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) const override {
        return inOut[pos].format == TensorFormat::kLINEAR && inOut[pos].type == DataType::kFLOAT;
    }

    const char* getPluginType() const override {
        return "HSwishPlugin";
    }

    const char* getPluginVersion() const override {
        return "1";
    }

    void configurePlugin(const DynamicPluginTensorDesc* in, int nbInputs, const DynamicPluginTensorDesc* out, int nbOutputs) override {}

    void attachToContext(cudnnContext* cudnn, cublasContext* cublas, IGpuAllocator* allocator) override {}

    void detachFromContext() override {}
};

class HSwishPluginCreator : public IPluginCreator {
public:
    const char* getPluginName() const override {
        return "HSwishPlugin";
    }

    const char* getPluginVersion() const override {
        return "1";
    }

    const PluginFieldCollection* getFieldNames() override {
        return nullptr;
    }

    IPluginV2* createPlugin(const char* name, const PluginFieldCollection* fc) override {
        return new HSwishPlugin();
    }

    IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) override {
        return new HSwishPlugin(serialData, serialLength);
    }

    void setPluginNamespace(const char* libNamespace) override {}

    const char* getPluginNamespace() const override {
        return "";
    }
};

REGISTER_TENSORRT_PLUGIN(HSwishPluginCreator);

extern "C" bool initLibNvInferPlugins(void* logger, const char* libNamespace) {
    return true;
}
