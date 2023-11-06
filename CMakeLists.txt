cmake_minimum_required(VERSION 2.6)
# set (CMAKE_C_COMPILER "/usr/bin/gcc")
# set (CMAKE_CXX_COMPILER "/usr/bin/g++")
project(demo_trt)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Debug)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/workspace)
set(HAS_PYTHON OFF)
# 修改为本机地址
set(TENSORRT_DIR "/home/.../TensorRT-8.4.0.6")
find_package(CUDA REQUIRED)
find_package(OpenCV)
include_directories(
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/src/trt
    ${PROJECT_SOURCE_DIR}/src/trt/common
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_TOOLKIT_ROOT_DIR}/include
    ${TENSORRT_DIR}/include
    ${CUDNN_DIR}/include
)
link_directories(
    ${TENSORRT_DIR}/lib
    ${CUDA_TOOLKIT_ROOT_DIR}/lib64
    ${CUDNN_DIR}/lib
)

set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -O0 -Wfatal-errors -pthread -w -g")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11 -O0 -Xcompiler -fPIC -g -w ${CUDA_GEN_CODE}")
file(GLOB_RECURSE cpp_srcs ${PROJECT_SOURCE_DIR}/src/*.cpp)
file(GLOB_RECURSE cuda_srcs ${PROJECT_SOURCE_DIR}/src/*.cu)
cuda_add_library(plugin_list SHARED ${cuda_srcs})
target_link_libraries(plugin_list nvinfer nvinfer_plugin)
target_link_libraries(plugin_list cuda cublas cudart cudnn)
target_link_libraries(plugin_list pthread)
target_link_libraries(plugin_list ${OpenCV_LIBS})
add_executable(demo_infer ${cpp_srcs})
target_link_libraries(demo_infer plugin_list)
