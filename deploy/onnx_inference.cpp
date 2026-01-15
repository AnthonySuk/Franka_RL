#include "onnx_inference.h"
#include <iostream>

OnnxInference::OnnxInference(const std::string& model_path, size_t input_size, size_t output_size)
    : env(ORT_LOGGING_LEVEL_WARNING, "test"), input_size_(input_size), output_size_(output_size) {
    Ort::SessionOptions session_options;
    session = new Ort::Session(env, model_path.c_str(), session_options);

    input_name = session->GetInputName(0, allocator);
    output_name = session->GetOutputName(0, allocator);
}

OnnxInference::~OnnxInference() {
    delete session;
}

std::vector<float> OnnxInference::update(const std::vector<float>& input_data) {
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    // Convert input_size_ to int64_t
    int64_t input_size_int64 = static_cast<int64_t>(input_size_);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, const_cast<float*>(input_data.data()), input_size_, &input_size_int64, 1);

    const char* input_names[] = {input_name};
    const char* output_names[] = {output_name};
    std::vector<Ort::Value> input_tensors;
    input_tensors.push_back(std::move(input_tensor));
    auto output_tensors = session->Run(Ort::RunOptions{nullptr}, input_names, input_tensors.data(), 1, output_names, 1);

    std::vector<float> output(output_size_);
    float* output_tensor_data = output_tensors[0].GetTensorMutableData<float>();
    for (size_t i = 0; i < output_size_; ++i) {
        output[i] = output_tensor_data[i];
    }

    return output;
}
