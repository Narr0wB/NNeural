
#ifndef MEMORY_H
#define MEMORY_H

#include <memory>
#define __HIP_PLATFORM_NVIDIA__
// #include <hip/hip_common.h>
// #include <hip/hip_runtime.h>
// #include <cuda.h>
// #include <thrust/device_vector.h>
// #include <cuda_runtime.h>

#include "Hardware.h"
#include "Log.h"

namespace memory {

// template <typename T>
// class buffer {
//     private:
//         cl::Device m_CLDevice;
//         cl::Buffer m_CLBuffer;

//         bool m_OwnedMemory;

//     public:
//         // The parameter <size> refers to the number of T elemets to be stored in the buffer 
//         buffer(cl::Device d, size_t size) : m_CLDevice(d), m_CLBuffer(cl::Context(d), size * sizeof(T)) {

//         }

//         buffer(size_t size) {}

//         T() const {
            
//         }
// };

// Define the custon GPU allocator and deleter
template <typename T>
struct gpu_deleter {
    void operator() (T* p) const noexcept {
        // if (hipFree(p) != hipSuccess) {
        //     LOG_ERROR("(GPU Deleter) Could not free gpu memory HIP ERROR: {}", hipGetLastError());
        // }
    }
};

template <typename T>
struct gpu_allocator {
    static T* allocate(size_t size) noexcept {
        if (size % sizeof(T) != 0) {
            LOG_ERROR("(GPU Allocator) Could not allocate memory (incomatible allocation size and T size)");
        }

        T* _ptr;

        // if (hipMalloc<T>(&_ptr, size) != hipSuccess) {
        //     LOG_ERROR("(GPU Allocator) Could not allocate memory HIP ERROR: {}", hipGetLastError());
        // }
    }
};


// Define the function that will allocate memory based on the hardware template argument
template <typename T, typename... Args>
std::shared_ptr<T> _make_shared(Args&&... args) {
    return std::make_shared<T>(std::forward<Args>(args)...);
    // switch (H) {
    //     case Hardware::CPU: {
            
    //         break;
    //     }

    //     case Hardware::GPU: {
    //         // Forward the parameters needed to construct T to the allocator (most likely a size_t)
    //         return std::shared_ptr<T>(gpu_allocator<T>::allocate(std::forward<Args>(args)...), gpu_deleter<T>());

    //         break;
    //     }

    //     default:
    //         return std::shared_ptr<T>();
    // }
}

}

#endif // MEMORY_H