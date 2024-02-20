
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

namespace Memory {

template <typename T>
class Buffer {
    private:
        bool m_MemOwned;
        bool m_HostOwnsDeviceMem;

        // Host side memory
        T* m_HBuffer;
        bool m_HMemOwned;

        // Device side memory
        Device m_Device;
        cl::Buffer m_CLBuffer;
        bool m_DMemOwned;

        size_t m_BufferSize;

        template <typename G>
        struct BufferIterator {
            using iterator_category = std::bidirectional_iterator_tag;                      
            using difference_type = std::ptrdiff_t;
            using value_type = G;
            using host_pointer = G*;
            using dev_pointer = std::uintptr_t;
            using reference = G&;

            private:
                host_pointer _ptr;

                BufferIterator(pointer p, difference_type off) : _ptr(p + off) {};

            public:
                refernce operator*() const { return *_ptr; }   
                host_pointer operator->() { return _ptr; }    

                BufferIterator& operator++() { _ptr++; return *this; }
                BufferIterator& operator--() { _ptr--; return *this; }
                BufferIterator& operator+=(difference_type offset) { _ptr += offset; return *this; }
                BufferIterator& operator-=(difference_type offset) { _ptr -= offset; return *this; }
                BufferIterator operator+(difference_type offset) { return BufferIterator(_ptr + offset); }
                BufferIterator operator-(difference_type offset) { return BufferIterator(_ptr - offset); }

                friend bool operator==(const BufferIterator& a, const BufferIterator& b) { return a._ptr == b._ptr; }  
                friend bool operator!=(const BufferIterator& a, const BufferIterator& b) { return a._ptr != b._ptr; } 
                friend BufferIterator operator-(const BufferIterator& a, const BufferIterator& b) { static_assert(a._ptr > b._ptr); return a._ptr - b._ptr; }
        };

    public:

        // The parameter <size> refers to the number of T elemets to be stored in the buffer
        Buffer(size_t size, cl::Device d) : 
        m_BufferSize(size * sizeof(T)), 
        m_CLBuffer(cl::Context(d), size * sizeof(T)),
        m_DMemOwned(true),
        m_HMemOwned(false) {
            m_HBuffer = (T*) m_Device.GetQueue().enqueueMapBuffer(m_CLBuffer, true, CL_MEM_ALLOC_HOST_PTR, 0, size * sizeof(T));
        }

        Buffer(size_t size) : 
        m_BufferSize(size * sizeof(T)),
        m_HBuffer(new T[size]),
        m_HMemOwned(true) {
        }

        // Create sub buffer that does NOT own the underlying memory
        // Buffer(T* host_ptr, size_t size) :
        // m_BufferSize(size * sizeof(T)),
        // m_HBuffer(host_ptr),
        // m_HMemOwned(false), {
        // }

        // Create a sub buffer which does not own the memory from a slice of another buffer 
        Buffer(const BufferIterator<T>& begin, const BufferIterator<T>& end) :
        m_BufferSize((begin - end) * sizeof(T)),
        m_HMemOwned(false),
        m_DMemOwned(false) {
            if (m_HBuffer != NULL) {
                
            }

        }

        ~Buffer() {
            if (m_MemOwned) {
                if (m_HBuffer != NULL)
                    delete m_HBuffer;
            }
        }
};

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