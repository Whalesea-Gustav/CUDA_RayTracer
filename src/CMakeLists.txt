cmake_minimum_required(VERSION 3.14)
project(test)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

SET(PROJECT_NAME test_src)

project(${PROJECT_NAME})



add_executable(${PROJECT_NAME}
        main.cu
        DeviceManager.cu DeviceManager.cu window.cpp scene.h camera.h util/float3Extension.cuh scene.cpp util/macros.h)

target_link_libraries(${PROJECT_NAME} PRIVATE header_lib)

set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_SEPARABLE_COMPILATION ON)