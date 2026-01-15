#include <iostream>
#include <fstream>
#include <vector>
#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>

// 简单的日志记录器
class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity != Severity::kINFO) {
            std::cout << "TensorRT Logger: " << msg << std::endl;
        }
    }
};

// 从文件中读取内容
std::vector<char> read_engine_file(const std::string& file_path) {
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

int main() {
    Logger logger;
    // 创建 TensorRT 运行时
    auto runtime = nvinfer1::createInferRuntime(logger);
    if (!runtime) {
        std::cerr << "Failed to create TensorRT runtime." << std::endl;
        return -1;
    }

    // 读取 .engine 文件
    std::string engine_file_path = "/home/lkx/Documents/gym/pointfoot-legged-gym/logs/tita_flat/exported/policies/model.engine";
    std::vector<char> engine_data = read_engine_file(engine_file_path);
    if (engine_data.empty()) {
        runtime->destroy();
        return -1;
    }

    // 反序列化引擎
    auto engine = runtime->deserializeCudaEngine(engine_data.data(), engine_data.size(), nullptr);
    if (!engine) {
        std::cerr << "Failed to deserialize TensorRT engine." << std::endl;
        runtime->destroy();
        return -1;
    }

    // 创建执行上下文
    auto context = engine->createExecutionContext();
    if (!context) {
        std::cerr << "Failed to create TensorRT execution context." << std::endl;
        engine->destroy();
        runtime->destroy();
        return -1;
    }

    // 准备输入数据
    size_t input_tensor_size = 32;
    std::vector<float> input_tensor_values(input_tensor_size);
    for (size_t i = 0; i < input_tensor_size; ++i) {
        input_tensor_values[i] = static_cast<float>(i);
    }

    // 分配 GPU 内存
    void* d_input;
    cudaMalloc(&d_input, input_tensor_size * sizeof(float));
    cudaMemcpy(d_input, input_tensor_values.data(), input_tensor_size * sizeof(float), cudaMemcpyHostToDevice);

    // 分配输出内存
    size_t output_tensor_size = 8;
    std::vector<float> output_tensor_values(output_tensor_size);
    void* d_output;
    cudaMalloc(&d_output, output_tensor_size * sizeof(float));

    // 设置输入输出绑定
    void* bindings[] = {d_input, d_output};

    // 运行推理
    context->enqueueV2(bindings, 0, nullptr);

    // 将结果从 GPU 复制到 CPU
    cudaMemcpy(output_tensor_values.data(), d_output, output_tensor_size * sizeof(float), cudaMemcpyDeviceToHost);

    // 打印输出结果
    for (size_t i = 0; i < output_tensor_size; ++i) {
        std::cout << "Output[" << i << "]: " << output_tensor_values[i] << std::endl;
    }

    // 释放资源
    cudaFree(d_input);
    cudaFree(d_output);
    context->destroy();
    engine->destroy();
    runtime->destroy();

    return 0;
}
