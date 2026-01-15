#include "tensorrt_inference.h"
#include <iostream>
#include <fstream>

void TensorRTInference::Logger::log(Severity severity, const char* msg) noexcept {
    if (severity != Severity::kINFO) {
        std::cout << "TensorRT Logger: " << msg << std::endl;
    }
}

std::vector<char> TensorRTInference::read_engine_file(const std::string& file_path) {
    std::ifstream file(file_path, std::ios::binary);
    if (!file.good()) {
        std::cerr << "Failed to open engine file: " << file_path << std::endl;
        return {};
    }
    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buffer(size);
    file.read(buffer.data(), size);
    file.close();
    return buffer;
}

TensorRTInference::TensorRTInference(const std::string& engine_path, size_t input_size, size_t output_size)
    : input_size_(input_size), output_size_(output_size) {
    runtime = nvinfer1::createInferRuntime(logger);
    std::vector<char> engine_data = read_engine_file(engine_path);
    engine = runtime->deserializeCudaEngine(engine_data.data(), engine_data.size(), nullptr);
    context = engine->createExecutionContext();

    cudaMalloc(&d_input, input_size_ * sizeof(float));
    cudaMalloc(&d_output, output_size_ * sizeof(float));
}

TensorRTInference::~TensorRTInference() {
    cudaFree(d_input);
    cudaFree(d_output);
    context->destroy();
    engine->destroy();
    runtime->destroy();
}

std::vector<float> TensorRTInference::update(const std::vector<float>& input_data) {
    cudaMemcpy(d_input, input_data.data(), input_size_ * sizeof(float), cudaMemcpyHostToDevice);

    void* bindings[] = {d_input, d_output};
    context->enqueueV2(bindings, 0, nullptr);

    std::vector<float> output(output_size_);
    cudaMemcpy(output.data(), d_output, output_size_ * sizeof(float), cudaMemcpyDeviceToHost);

    return output;
}
