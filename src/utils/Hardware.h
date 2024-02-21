
#ifndef HARDWARE_H
#define HARDWARE_H 

#include <type_traits>
#include <vector>
#include <iostream>
#include <fstream>
#include <CL/cl.hpp>

#include "Log.h"
#include "Memory.h"

#define MAX_CPU_THREADS 8

enum class Hardware {
    CPU = 0,
    GPU,
};



// inline cl::Platform OpenCLInit() {
//     std::vector<cl::Platform> platforms;
//     cl::Platform::get(&platforms);

//     return platforms[0];
// }

// const cl::Platform default_platform = OpenCLInit();

inline cl::Device GetDefaultGPU() {
    std::vector<cl::Device> gpus;

    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &gpus);

    if (gpus.size() == 0) {
        return cl::Device();
    }

    return gpus[0];
}

inline cl::Device GetDefaultCPU() {
    std::vector<cl::Device> cpus;

    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &cpus);

    return cpus[0];
}

const cl::Device default_gpu = GetDefaultCPU();
const cl::Device default_cpu = GetDefaultGPU();

inline std::string load_kernel(const char* path) {
    std::ifstream source_file(path);

    if (not source_file.is_open()) {
        LOG_ERROR("Could not load OpenCL Kernel!");
    }

    auto temp = std::string(std::istreambuf_iterator<char>(source_file), std::istreambuf_iterator<char>());

    return temp;
}

// Enable function declaration based on template arguments
template <bool B>
using EnableIf = typename std::enable_if<B, int>::type;

#endif // HARDWARE_H