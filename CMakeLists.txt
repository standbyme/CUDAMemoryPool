cmake_minimum_required(VERSION 3.19)
project(CUDAMemoryPool CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(CUDAMemoryPool memory_pool.h main.cu )

set_target_properties(CUDAMemoryPool PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
