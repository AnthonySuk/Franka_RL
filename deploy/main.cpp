#include "onnx_inference.h"
#include "tensorrt_inference.h"
#include <iostream>
#include <string>
#include <vector>

struct ModelParams
{
    float damping;
    float stiffness;
    float action_scale;
    float hip_scale_reduction;
    float num_of_dofs;
    float lin_vel_scale;
    float ang_vel_scale;
    float dof_pos_scale;
    float dof_vel_scale;
    float clip_obs;
    float clip_actions;
    float torque_limits[8];
    float d_gains[8];
    float p_gains[8];
    float commands_scale[3];
    float default_dof_pos[8];
};

struct Observations
{
    float lin_vel[3];
    float ang_vel[3];
    float gravity_vec[3];
    float forward_vec[3];
    float commands[3];
    float base_quat[4];
    float dof_pos[8];
    float dof_vel[8];
    float actions[8];
};

class InferenceRunner
{
public:
    // Constructor, initialize inference instances directly
    InferenceRunner()
    {
        std::string onnx_model_path = "/home/lkx/Documents/gym/pointfoot-legged-gym/logs/tita_flat/exported/policies/policy.onnx";
        std::string engine_path = "/home/lkx/Documents/gym/pointfoot-legged-gym/logs/tita_flat/exported/policies/model.engine";
        size_t input_size = 31;
        size_t output_size = 8;
        input_size_ = input_size;
        output_size_ = output_size;
        observations_.resize(input_size);
        actions_.resize(output_size);
        last_actions_.resize(output_size);
        last_actions_.assign(output_size, 0.0f);
        try {
            // Initialize ONNX and TensorRT inference instances
            onnx_inference_ = std::make_shared<OnnxInference>(onnx_model_path, input_size, output_size);
            tensorrt_inference_ = std::make_shared<TensorRTInference>(engine_path, input_size, output_size);
            std::cout << "Inference instances initialized successfully." << std::endl;
        }
        catch (const std::exception& e) {
            std::cerr << "Error initializing inference: " << e.what() << std::endl;
        }
    }
    // // Update function, used to run inference
    bool update()
    {
        computeObeservations();
        computeActions();

        return true;
    }
    void computeObeservations()
    {
        // 准备输入数据
        std::vector<float> input_data(input_size_);
        for (size_t i = 0; i < input_size_; ++i)
        {
            observations_[i] = static_cast<float>(i);
        }
    }
    void computeActions() {
        // 运行 ONNX 推理
        std::vector<float> onnx_output = onnx_inference_->update(observations_);
        std::cout << "ONNX Output:" << std::endl;
        for (size_t i = 0; i < output_size_; ++i)
        {
            std::cout << "Output[" << i << "]: " << onnx_output[i] << std::endl;
        }

        // 运行 TensorRT 推理
        std::vector<float> tensorrt_output = tensorrt_inference_->update(observations_);
        std::cout << "TensorRT Output:" << std::endl;
        for (size_t i = 0; i < output_size_; ++i)
        {
            std::cout << "Output[" << i << "]: " << tensorrt_output[i] << std::endl;
        }

    }

private:
    // ModelParams params_;
    // Observations obs_;
    std::string onnx_model_path_;
    std::string engine_path_;
    size_t input_size_;
    size_t output_size_;
    std::vector<float> observations_;
    std::vector<float> actions_, last_actions_;
    std::shared_ptr<OnnxInference> onnx_inference_;
    std::shared_ptr<TensorRTInference> tensorrt_inference_;
};

int main()
{

    // Create an InferenceRunner instance
    InferenceRunner runner;

    // Run the inference process
    return runner.update();
}
