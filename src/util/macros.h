#pragma once

#include <cuda_runtime_api.h>
#include <stdint.h>
#include "logger.h"

// http://www.decompile.com/cpp/faq/file_and_line_error_string.htm
#define STRINGIFY(x) TOSTRING(x)
#define TOSTRING(x) #x

// CUDA_ERROR
inline void HandleError(cudaError_t err, const char* file, int line)
{
    // Error handling micro, wrap it around function whenever possible
    if (err != cudaSuccess) {
        gprender::GPLog::get_logger()->error("Line {} File {}", line, file);
        gprender::GPLog::get_logger()->error("CUDA ERROR: {}", cudaGetErrorString(err));
#ifdef _WIN32
        system("pause");
#else
        exit(EXIT_FAILURE);
#endif
    }
}

#define CUDA_ERROR(err) (::gpmesh::HandleError(err, __FILE__, __LINE__))

#define M_PI 3.1415926535897932384626433832795
#define EPSILON 0.001
#define HIT_EPSILON 1e-4f

// unsigned 64-bit
#define INVALID64 0xFFFFFFFFFFFFFFFFu

// unsigned 32-bit
#define INVALID32 0xFFFFFFFFu

// unsigned 16-bit
#define INVALID16 0xFFFFu