# cmake_minimum_required(VERSION 3.10)

# # 设置项目名称
# project(OnnxInference)

# # 设置 C++ 标准
# set(CMAKE_CXX_STANDARD 11)

# # 查找 ONNX Runtime 库
# # find_package(ONNXRuntime REQUIRED)

# # 添加可执行文件
# add_executable(${PROJECT_NAME} main.cpp)

# # 链接 ONNX Runtime 库
# # target_link_libraries(onnx_inference PRIVATE ONNXRuntime::ONNXRuntime)
# target_link_libraries(${PROJECT_NAME}
#   onnxruntime          
# )

###########################
# 设置 CMake 最低版本要求
cmake_minimum_required(VERSION 3.10)

# 设置项目名称
project(InferenceProjects)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 11)
include_directories(/usr/local/TensorRT-8.6.1.6/include)
link_directories(/usr/local/TensorRT-8.6.1.6/lib)
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

add_executable(main main.cpp onnx_inference.cpp tensorrt_inference.cpp)
target_link_libraries(main PRIVATE onnxruntime cudart nvinfer)