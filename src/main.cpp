
#include <iostream>
#include "tensor/Tensor.h"
#include "tensor/TensorOperations.h"
#include "utils/Log.h"
#include "model/Layer.h"

template <typename T>
T ReLU(T in) {
    return in > 0 ? in : 0;
}

// TODO:
// Implement the gpu kernels for forward and backward propagation
// Check performance for host allocated device accessible memory (hipHostMalloc)

int main(void) {
    Log::Init();
    
    // Tensor<FP32> t = {
    //     {2, 3, 4},
    //     {1, 1, 1},
    //     {1, 1, 1}
    // };
    
    // Simple vector add
    LOG_INFO("Using device: ");
    std::cout << default_gpu.getInfo<CL_DEVICE_NAME>() << std::endl;

    cl::Context ctx(default_gpu);

    cl::Program::Sources source;

    auto src = load_kernel("../test.cl");
    source.push_back({src.c_str(), src.size()});

    cl::Program program(ctx, source);
    if (program.build({default_gpu}) != CL_SUCCESS) {
        LOG_ERROR("Could not compile kernel");
    }

    int N = 1000000;

    cl::Buffer a_buffer(ctx, CL_MEM_READ_WRITE, sizeof(int) * N);
    cl::Buffer b_buffer(ctx, CL_MEM_READ_WRITE, sizeof(int) * N);
    cl::Buffer c_buffer(ctx, CL_MEM_READ_WRITE, sizeof(int) * N);

    int* A = new int[N];
    int* B = new int[N];
    for (int i = 0; i < N; ++i) {
        A[i] = 5;
        B[i] = 3;
    }

    cl::CommandQueue queue(ctx, default_gpu);

    queue.enqueueWriteBuffer(a_buffer, CL_TRUE, 0, sizeof(A[0]) * N, A);
    queue.enqueueWriteBuffer(b_buffer, CL_TRUE, 0, sizeof(B[0]) * N, B);

    cl::Kernel simple_add(program, "test_kernel");
    simple_add.setArg(0, a_buffer);
    simple_add.setArg(1, b_buffer);
    simple_add.setArg(2, c_buffer);
    simple_add.setArg(3, sizeof(int), &N);

    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < 100; ++i) queue.enqueueNDRangeKernel(simple_add, cl::NullRange, cl::NDRange(1024), cl::NullRange);

    int* C = new int[N];
    queue.enqueueReadBuffer(c_buffer, CL_TRUE, 0, sizeof(int) * N, C);
    auto stop = std::chrono::high_resolution_clock::now();

    // std::cout << "result: {";
    // for (int i = 0; i < N; i++) {
    //     std::cout << C[i] << " ";
    // }
    // std::cout << "}" << std::endl;
    LOG_INFO("It took {}us to execute the kernel", std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count());

    return 0;
}