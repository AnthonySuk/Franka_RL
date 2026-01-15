#ifndef ONNX_INFERENCE_H
#define ONNX_INFERENCE_H

#include <onnxruntime_cxx_api.h>
#include <vector>

class OnnxInference {
public:
    OnnxInference(const std::string& model_path, size_t input_size, size_t output_size);
    ~OnnxInference();
    std::vector<float> update(const std::vector<float>& input_data);

private:
    Ort::Env env;
    Ort::Session* session;
    size_t input_size_;
    size_t output_size_;
    Ort::AllocatorWithDefaultOptions allocator;
    const char* input_name;
    const char* output_name;
};

#endif // ONNX_INFERENCE_H
