cmake_minimum_required(VERSION 3.22)
project(CUDA_RayTracer LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

set(${EXECUTABLE_OUTPUT_PATH} "${PROJECT_SOURCE_DIR}")
set(${LIBRARY_OUTPUT_PATH} "${PROJECT_SOURCE_DIR}")

include(FetchContent)

# spdlog
FetchContent_Declare(spdlog
        GIT_REPOSITORY https://github.com/gabime/spdlog.git
        GIT_TAG        v1.8.5
        )
FetchContent_Populate(spdlog)

# Auto-detect GPU architecture, sets ${CUDA_ARCHS}
include("cmake/AutoDetectCudaArch.cmake")

# CUDA and C++ compiler flags
set(cxx_flags
        $<$<CXX_COMPILER_ID:MSVC>:-D_SCL_SECURE_NO_WARNINGS /openmp /std:c++17>           #Add MSVC-specific compiler flags here
        #	$<$<CXX_COMPILER_ID:GNU>: -fopenmp -std=c++17> #Add GCC/Clang-specific compiler flags here
        $<$<CXX_COMPILER_ID:GNU>:-Wall -m64 -fopenmp -O3 -std=c++17 -Wno-unused-function> #Add GCC/Clang-specific compiler flags here
        )
set(cuda_flags
        -Xcompiler=$<$<CXX_COMPILER_ID:GNU>:-Wall -O3>
        #Disables warning
        #177-D "function XXX was declared but never referenced"
        -Xcudafe "--display_error_number --diag_suppress=177"
        ${CUDA_ARCHS}
        -rdc=true
        -lineinfo
        --expt-extended-lambda
        #-use_fast_math
        $<$<CXX_COMPILER_ID:GNU>:-O3>
        --expt-relaxed-constexpr
        -Xptxas -warn-spills -res-usage
        --ptxas-options=-v
        -G
        )


add_library(developer_flags INTERFACE)
target_compile_options(developer_flags INTERFACE
        $<$<COMPILE_LANGUAGE:CXX>:${cxx_flags}>
        $<$<COMPILE_LANGUAGE:CUDA>:${cuda_flags}>
        )

# RXMesh: could think of this as just the header library, so name header_lib
add_library(header_lib INTERFACE)
target_compile_features(header_lib INTERFACE cxx_std_17)
set(CMAKE_CXX_EXTENSIONS OFF)
target_compile_definitions(header_lib
        INTERFACE INPUT_DIR=${CMAKE_CURRENT_SOURCE_DIR}/input/
        INTERFACE OUTPUT_DIR=${CMAKE_CURRENT_SOURCE_DIR}/output/
        )

target_link_libraries(header_lib INTERFACE $<BUILD_INTERFACE:developer_flags>)

target_include_directories( header_lib
        INTERFACE "include"
        INTERFACE "include/glm"
        INTERFACE "${spdlog_SOURCE_DIR}/include"
        INTERFACE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
        )

#OpenMP
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    target_link_libraries(header_lib INTERFACE OpenMP::OpenMP_CXX)
endif()

add_subdirectory(cpp_version)
add_subdirectory(src)

add_executable(CUDA_RayTracer main.cpp)
