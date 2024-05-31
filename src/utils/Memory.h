
#ifndef MEMORY_H
#define MEMORY_H

#include <memory>
// #include <hip/hip_common.h>
// #include <hip/hip_runtime.h>
// #include <cuda.h>
// #include <thrust/device_vector.h>
// #include <cuda_runtime.h>
#include "Log.h"
#include "Hardware.h"

class Device {
    private:
        cl::Device _device;
        cl::Context _device_context;
        cl::CommandQueue _device_queue;

    public:

        Device(cl::Device d) : 
        _device(d),
        _device_context(d) {
            _device_queue = cl::CommandQueue(_device_context, d);
        }

        Device(cl::Device d, cl::Context c, cl::CommandQueue cq) :
        _device(d),
        _device_context(c),
        _device_queue(cq) {

        }

        Device(const Device& other) :
        _device(other._device),
        _device_context(other._device_context),
        _device_queue(other._device_queue) {

        }

        Device() {};

        inline const cl::CommandQueue& GetQueue() const { return _device_queue; };
        inline const cl::Context& GetContext() const { return _device_context; }
        friend bool operator==(const Device& a, const Device& b) { return a._device() == b._device(); }
        friend bool operator!=(const Device& a, const Device& b) { return a._device() != b._device(); }
};

#define NULL_DEVICE Device()

namespace Memory {

template <typename T>
class Buffer {
    private:
        bool m_MemOwned;
        size_t m_Size;

        // Host side memory
        T* m_HBuffer;

        // Device side memory
        Device m_Device;
        cl::Buffer m_CLBuffer;


    public:

        struct BufferIterator {
            using iterator_category = std::bidirectional_iterator_tag;                      
            using difference_type = std::ptrdiff_t;
            using value_type = T;
            using host_pointer = T*;
            using dev_data = cl::Buffer;
            using dev_pointer = std::uintptr_t;
            using reference = T&;

            #define INVALID (std::uintptr_t)-1

            private:
                host_pointer _ptr;
                dev_pointer _dev_ptr;
                dev_data _dev_data;

            public:
                // BufferIterator(host_pointer p, difference_type off) : _ptr(p + off) {};
                BufferIterator(host_pointer p) : _ptr(p), _dev_ptr(INVALID), _dev_data() {}
                // BufferIterator(dev_pointer d_p, difference_type off) : _ptr(NULL), _dev_ptr(d_p + off) {}
                BufferIterator(dev_pointer d_p, dev_data& buf_ref) : _ptr(NULL), _dev_ptr(d_p), _dev_data(buf_ref) {}
                // BufferIterator(host_pointer p, dev_pointer d_p, difference_type off) : _ptr(p + off), _dev_ptr(d_p + off) {}
                BufferIterator(host_pointer p, dev_pointer d_p, dev_data& buf_ref) : _ptr(p), _dev_ptr(d_p), _dev_data(buf_ref) {}

                reference operator*() const { assert(_ptr != NULL); return *_ptr; }   
                operator size_t() const { return (size_t)(10); }
                //host_pointer operator->() { assert(_ptr != NULL); return _ptr; } 

                dev_pointer device_ptr() const { return _dev_ptr; }
                dev_data device_data() const { return _dev_data; }
                host_pointer host_ptr() const { return _ptr; }

                operator void*() {
                    assert(_ptr != NULL);

                    return (void*) _ptr;
                }

                operator T*() {
                    assert(_ptr != NULL);

                    return _ptr;
                }

                BufferIterator& operator++() { _ptr++; if (_dev_ptr != INVALID) _dev_ptr++; return *this; }
                BufferIterator& operator--() { _ptr--; _dev_ptr--; return *this; }
                BufferIterator& operator+=(difference_type offset) { _ptr += offset; _dev_ptr += offset; return *this; }
                BufferIterator& operator-=(difference_type offset) { _ptr -= offset; _dev_ptr -= offset; return *this; }
                BufferIterator operator+(difference_type offset) { BufferIterator temp = *this; temp._ptr += offset; temp._dev_ptr += offset; return temp; }
                BufferIterator operator-(difference_type offset) { BufferIterator temp = *this; temp._ptr -= offset; temp._dev_ptr += offset; return temp; }

                friend bool operator==(const BufferIterator& a, const BufferIterator& b) { return a._ptr == b._ptr && a._dev_ptr == b._dev_ptr; }  
                friend bool operator!=(const BufferIterator& a, const BufferIterator& b) { return a._ptr != b._ptr || a._dev_ptr != b._dev_ptr; } 
                friend BufferIterator operator-(const BufferIterator& a, const BufferIterator& b) { assert(a._ptr >= b._ptr); return BufferIterator((host_pointer)(uintptr_t)(a._ptr - b._ptr)); }
        };

        // The parameter <size> refers to the number of T elemets to be stored in the buffer
        Buffer(size_t size, Device d) : 
        m_Size(size), 
        m_Device(d),
        m_CLBuffer(d.GetContext(), CL_MEM_READ_WRITE, size * sizeof(T)),
        m_MemOwned(true), 
        m_HBuffer(NULL) {
        }

        Buffer(T* host_data, size_t size, Device d) : 
        m_Size(size), 
        m_Device(d),
        m_CLBuffer(d.GetContext(), CL_MEM_READ_WRITE, size * sizeof(T)),
        m_MemOwned(true),
        m_HBuffer(NULL) {
            assert(host_data != NULL);

            d.GetQueue().enqueueWriteBuffer(m_CLBuffer, CL_TRUE, 0, size * sizeof(T), (void*) host_data);
        }

        Buffer(size_t size) : 
        m_Size(size),
        m_HBuffer(new T[size]),
        m_MemOwned(true) {
        }

        // Create sub buffer that does NOT own the underlying memory
        // Buffer(T* host_ptr, size_t size) :
        // m_Size(size * sizeof(T)),
        // m_HBuffer(host_ptr),
        // m_HMemOwned(false), {
        // }

        // Create a sub buffer which does not own the memory from a slice of another buffer 
        Buffer(const BufferIterator& begin, const BufferIterator& end) :
        m_Size((end - begin)),
        m_MemOwned(false) {
            m_HBuffer = begin.host_ptr();
        }

        Buffer(T val) :
        m_Size(1),
        m_HBuffer(new T[1]), 
        m_MemOwned(true) {
            m_HBuffer[0] = val;
        }

        Buffer(Buffer<T>&& b) :
        m_Size(b.m_Size), 
        m_Device(b.m_Device),
        m_CLBuffer(b.m_CLBuffer),
        m_MemOwned(b.m_MemOwned),
        m_HBuffer(b.m_HBuffer) {
            b.m_MemOwned = false;
        }

        Buffer(const Buffer<T>& b) :
        m_Size(b.m_Size), 
        m_Device(b.m_Device),
        m_CLBuffer(b.m_CLBuffer),
        m_MemOwned(false),
        m_HBuffer(b.m_HBuffer) {}

        Buffer& operator=(Buffer&& other) {
            m_Size = other.m_Size;
            m_Device = other.m_Device;
            m_CLBuffer = other.m_CLBuffer;
            m_MemOwned = other.m_MemOwned;
            m_HBuffer = other.m_HBuffer;

            other.m_MemOwned = false; // This is why I fucking hate this language

            return *this;
        }

        Buffer& operator=(const Buffer& other) {
            m_Size = other.m_Size;
            m_Device = other.m_Device;
            m_CLBuffer = other.m_CLBuffer;
            m_MemOwned = other.m_MemOwned;
            m_HBuffer = other.m_HBuffer;

            other.m_MemOwned = false;

            return *this;
        }

        Buffer() {};

        ~Buffer() {
            if (m_MemOwned) {
                if (m_HBuffer != NULL) {
                    delete m_HBuffer;
                }
            }
        }

        inline cl::Buffer GetDeviceBuffer() const { return m_CLBuffer; }
        inline T* GetHostBuffer() const { return m_HBuffer; }
        inline size_t GetSize() const { return m_Size; }
        inline BufferIterator begin() { 
            if (m_HBuffer != NULL && m_Device != NULL_DEVICE) {
                return BufferIterator(m_HBuffer, 0, m_CLBuffer); 
            }
            else if (m_HBuffer != NULL) {
                return BufferIterator(m_HBuffer);
            }
            else {
                return BufferIterator(0, m_CLBuffer);
            }
        }

        inline BufferIterator end() { 
            if (m_HBuffer != NULL && m_Device != NULL_DEVICE) {
                return BufferIterator((T*)(m_HBuffer + m_Size), m_Size, m_CLBuffer); 
            }
            else if (m_HBuffer != NULL) {
                return BufferIterator((T*)(m_HBuffer + m_Size));
            }
            else {
                return BufferIterator(m_Size, m_CLBuffer);
            }
        }

        void copy(const Device& d) {
            if (!m_MemOwned) { LOG_ERROR("Buffers that do not own their memory cannot be copied!"); }

            // if we pass the NULL DEVICE it means we want to map TO host (it is implied that we are dealing with a device buffer)
            if (d == NULL_DEVICE) {
                assert(m_Device != NULL_DEVICE);

                if (m_HBuffer == NULL) {
                    m_HBuffer = new T[m_Size];
                } 

                m_Device.GetQueue().enqueueReadBuffer(m_CLBuffer, true, 0, m_Size * sizeof(T), m_HBuffer);

                return;
            }

            assert(m_Device == NULL_DEVICE);

            m_Device = d;
            m_CLBuffer = cl::Buffer(d.GetContext(), CL_MEM_READ_WRITE, m_Size * sizeof(T));
            d.GetQueue().enqueueWriteBuffer(m_CLBuffer, CL_TRUE, 0, m_Size * sizeof(T), m_HBuffer);
        }

        void delete_copy() {
            if (m_Device == NULL_DEVICE) {
                m_CLBuffer = nullptr;

                return;
            }

            if (m_HBuffer != NULL) {
                delete m_HBuffer;
            }
        }

        cl::Buffer getTemp() { return m_CLBuffer; }

        // void unmap() {
        //     // Unmap a previous device TO host mapping
        //     if (m_Device == NULL_DEVICE) {
        //         assert(m_HBuffer != NULL && m_HostMemOwnedByMapping == true);

        //         m_Device.GetQueue().enqueueUnmapMemObject(m_CLBuffer, m_HBuffer);
        //         m_HostMemOwnedByMapping = false;
        //         m_HBuffer = NULL;

        //         return;
        //     }

        //     m_CLBuffer = nullptr;
        //     m_Device = NULL_DEVICE;
        // }

        void move(const Device& d) {
            if (!m_MemOwned) { LOG_ERROR("Buffers that do not own their memory cannot be moved!"); }

            // Move TO host (NULL_DEVICE)
            if (d == NULL_DEVICE) {
                // Make sure we are not moving a host buffer to host again
                assert(m_Device != NULL_DEVICE);

                
                if (m_HBuffer == NULL) {
                    m_HBuffer = new T[m_Size];
                }
                
                // Move the data
                m_Device.GetQueue().enqueueReadBuffer(m_CLBuffer, true, 0, m_Size * sizeof(T), m_HBuffer);
                m_CLBuffer = nullptr;
                m_Device = d;

                return;
            }

            
            // Make sure we are not moving the buffer to the same device
            assert(m_Device != d);

            // If we are moving a host buffer to a device
            if (m_Device == NULL_DEVICE) {
                // Make sure we have data in the host buffer
                assert(m_HBuffer != NULL);

                m_CLBuffer = cl::Buffer(d.GetContext(), CL_MEM_READ_WRITE, m_Size * sizeof(T));
                d.GetQueue().enqueueWriteBuffer(m_CLBuffer, true, 0, m_Size * sizeof(T), m_HBuffer);
                m_Device = d;

                delete m_HBuffer;
                m_HBuffer = NULL;
            }
            
            // Device to device
            else {

                LOG_ERROR("Device to Device transfers still not implemented");

                // cl::Buffer new_dev_buf(d.GetContext(), CL_MEM_READ_WRITE, m_Size);
                // m_Device.GetQueue().enqueueCopyBuffer()
            }
        }

        operator cl::Buffer() const {
            assert(m_Device != NULL_DEVICE);

            return m_CLBuffer;
        }

        T operator[](size_t idx) const {
            assert(m_HBuffer != NULL);

            return m_HBuffer[idx];
        }

        T& operator[](size_t idx) {
            assert(m_HBuffer != NULL);

            return m_HBuffer[idx];
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