
#include <iostream>
#include "tensor/Tensor.h"
#include "tensor/TensorOperations.h"
#include "utils/Log.h"
#include "model/Layer.h"
#include "utils/Memory.h"

template <typename T>
T ReLU(T in) {
    return in > 0 ? in : 0;
}

// TODO:
// Implement the gpu kernels for forward and backward propagation
// Check performance for host allocated device accessible memory (hipHostMalloc)

int main(void) {
    Log::Init();
    
    Tensor<FP32> t = { 
        {
            {2, 3, 4},
            {1, 1, 1},
            {1, 1, 1}
        }, 
        {
            {2, 3, 4},
            {1, 1, 1},
            {1, 1, 1}
        }
    };

    Tensor<FP32> g = {
        1, 2, 3
    };

    Tensor<FP32> eddu = 3;

    std::cout << t[0][0] << std::endl;
    std::cout << g << std::endl;
    
    std::cout << add(t[0], eddu) << std::endl;
    
    std::cout << 2 + t[1][1][1] << std::endl;
    
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

    cl::CommandQueue queue(ctx, default_gpu);

    int N = 10;

    int G[] = { 10, 3, 5, 6, 7, 8, 9, 10 };

    Device gpu_device(default_gpu, ctx, queue);

    // Create host buffer
    Memory::Buffer<int> buffer((size_t) N);
    Memory::Buffer<int> buffer_device(G, sizeof(G) / sizeof(G[0]), gpu_device);

    // fill the buffer
    for (auto& i : buffer) {
        i = 10;
    }


    cl::Buffer a_buffer(ctx, CL_MEM_READ_WRITE, sizeof(int) * N);
    cl::Buffer b_buffer(ctx, CL_MEM_READ_WRITE, sizeof(int) * N);
    cl::Buffer c_buffer(ctx, CL_MEM_READ_WRITE, sizeof(int) * N);

    int* A = new int[N];
    int* B = new int[N];
    for (int i = 0; i < N; ++i) {
        A[i] = 5;
        B[i] = 3;
    }

    queue.enqueueWriteBuffer(a_buffer, CL_TRUE, 0, sizeof(A[0]) * N, A);
    queue.enqueueWriteBuffer(b_buffer, CL_TRUE, 0, sizeof(B[0]) * N, B);

    buffer.move(gpu_device);
    buffer_device.copy(NULL_DEVICE);

    for (int i = 0; i < 8; ++i) {
        LOG_WARN("WE COPIED {}", buffer_device[i]);
    }

    //LOG_INFO("se {}", *(buffer.begin()));

    cl::Kernel simple_add(program, "test_kernel");
    simple_add.setArg(0, a_buffer);
    simple_add.setArg(1, (cl::Buffer) buffer_device);
    simple_add.setArg(2, c_buffer);
    simple_add.setArg(3, sizeof(int), &N);

    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < 100; ++i) queue.enqueueNDRangeKernel(simple_add, cl::NullRange, cl::NDRange(10), cl::NullRange);

    int* C = new int[N];
    queue.enqueueReadBuffer(c_buffer, CL_TRUE, 0, sizeof(int) * N, C);
    auto stop = std::chrono::high_resolution_clock::now();

    std::cout << "result: {";
    for (int i = 0; i < N; i++) {
        std::cout << C[i] << " ";
    }
    std::cout << "}" << std::endl;
    LOG_INFO("It took {}us to execute the kernel", std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count());

    return 0;
}