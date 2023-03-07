#include "DeviceManager.h"
#include <cuda_runtime_api.h>
#include <cmath>
#include <cstdlib>
#include <curand_kernel.h>
#include <helper_cuda.h>
#include "util/logger.h"

void PrintDeviceInfo() {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);

    if (device_count == 0)
    {
        GPRENDER_ERROR("No CUDA devices found!");
        return;
    }

    for (auto dev = 0; dev < device_count; dev++)
    {
        cudaSetDevice(dev);
        cudaDeviceProp device_prop{};
        cudaGetDeviceProperties(&device_prop, dev);
        GPRENDER_INFO("Device {}: \"{}\"\n", dev, device_prop.name);
        GPRENDER_INFO("Global Memory Size: {} MBytes ({} bytes) ", static_cast<float>(device_prop.totalGlobalMem / 1048576.0f), static_cast<unsigned long long>(device_prop.totalGlobalMem));
        GPRENDER_INFO("SM number: {}", device_prop.multiProcessorCount);
        GPRENDER_INFO("CUDA cores per SM: {}", _ConvertSMVer2Cores(device_prop.major, device_prop.minor));
        GPRENDER_INFO("Total CUDA cores: {}", _ConvertSMVer2Cores(device_prop.major, device_prop.minor) * device_prop.multiProcessorCount);
        GPRENDER_INFO("Static Memory Size: {} bytes", device_prop.totalConstMem);
        GPRENDER_INFO("Shared Memory Size per Block: {} MBytes ({} bytes)", device_prop.sharedMemPerBlock / 1024.0f, device_prop.sharedMemPerBlock);
        GPRENDER_INFO("Registers per Block: {}", device_prop.regsPerBlock);
        GPRENDER_INFO("Warp Size: {}", device_prop.warpSize);
        GPRENDER_INFO("Max Threads per SM: {}", device_prop.maxThreadsPerMultiProcessor);
        GPRENDER_INFO("Max Threads per Block: {}", device_prop.maxThreadsPerBlock);
        GPRENDER_INFO("Max Threads Dimension: ({}, {}, {})",
                      device_prop.maxThreadsDim[0], device_prop.maxThreadsDim[1], device_prop.maxThreadsDim[2]);
        GPRENDER_INFO("Max Grid Size: ({}, {}, {})",
                      device_prop.maxGridSize[0], device_prop.maxGridSize[1], device_prop.maxGridSize[2]);
    }
    GPRENDER_INFO("Device Info Print Done!");
}
