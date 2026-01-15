#ifndef TENSORRT_INFERENCE_H
#define TENSORRT_INFERENCE_H

#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>
#include <vector>
#include <string> // Add this line

class TensorRTInference {
public:
    TensorRTInference(const std::string& engine_path, size_t input_size, size_t output_size);
    ~TensorRTInference();
    std::vector<float> update(const std::vector<float>& input_data);

private:
    class Logger : public nvinfer1::ILogger {
        void log(Severity severity, const char* msg) noexcept override;
    };

    Logger logger;
    nvinfer1::IRuntime* runtime;
    nvinfer1::ICudaEngine* engine;
    nvinfer1::IExecutionContext* context;
    size_t input_size_;
    size_t output_size_;
    void* d_input;
    void* d_output;
    std::vector<char> read_engine_file(const std::string& file_path);
};

#endif // TENSORRT_INFERENCE_H
