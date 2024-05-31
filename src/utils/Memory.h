
#ifndef MEMORY_H
#define MEMORY_H

#include <memory>
// #include <hip/hip_common.h>
// #include <hip/hip_runtime.h>
#include <cuda.h>
// #include <thrust/device_vector.h>
#include <cuda_runtime.h>
#include "Log.h"
#include "Hardware.h" 

enum Device {
  CPU, GPU
};

namespace Memory {

template <typename T>
class Buffer {
    private:
      bool m_MemOwned;
      Device m_Device = Device::CPU;
      
      // The size of a buffer is specified in number of T instances inside the buffer
      size_t m_Size;

      // Memory buffers
      T* m_Buffer;

    public:
      
      inline size_t size() {return m_Size;}
      inline Device device() {return m_Device;}
      inline T* buffer() {return m_Buffer;}
      
      // The parameter <size> refers to the number of T elemets to be stored in the buffer
      Buffer(size_t size, Device d = Device::CPU) : 
      m_Size(size), 
      m_Device(d),
      m_MemOwned(true), 
      m_Buffer(NULL) {
        switch (m_Device) {
          case Device::CPU: {
            m_Buffer = new T[size]; 
            break;
          }
          case Device::GPU: {
            m_Buffer = cudaMalloc(&m_Buffer, size * sizeof(T));
            break;
          }
        }
      }
      
      // WARNING: This constructor WILL take ownership of the pointer, be careful!
      Buffer(T* host_data, size_t size) : 
      m_Size(size), 
      m_Device(Device::CPU),
      m_MemOwned(true), 
      m_Buffer(host_data) {
      }

      Buffer(size_t size) : 
      m_Size(size),
      m_Device(Device::CPU),
      m_Buffer(new T[size]),
      m_MemOwned(true) {
      }

      // Create sub buffer that does NOT own the underlying memory
      // Buffer(T* host_ptr, size_t size) :
      // m_Size(size * sizeof(T)),
      // m_HBuffer(host_ptr),
      // m_HMemOwned(false), {
      // }

      // Create a sub buffer which does not own the memory from a slice of another buffer
      // begin, end: Are offsets (in terms of how many T) from the beginning of the lending buffer
      Buffer(const Buffer<T>& other, const size_t begin, const size_t end) :
      m_Size((end - bg egin)),
      m_Device(other.m_Device),
      m_MemOwned(false) {
        m_Buffer = (T*)( (uintptr_t)other.m_Buffer + begin );
      }

      // Move constructor 
      Buffer(Buffer<T>&& b) :
      m_Size(b.m_Size), 
      m_Device(b.m_Device),
      m_MemOwned(b.m_MemOwned),
      m_Buffer(b.m_Buffer) {
        b.m_MemOwned = false;
      }
      
      // Copy constructor
      Buffer(const Buffer<T>& b) :
      m_Size(b.m_Size), 
      m_Device(b.m_Device),
      m_MemOwned(false),
      m_Buffer(b.m_Buffer) {}
      
      // Move assignment operator
      Buffer& operator=(Buffer&& other) {
          m_Size = other.m_Size;
          m_Device = other.m_Device;
          m_MemOwned = other.m_MemOwned;
          m_Buffer = other.m_Buffer;

          other.m_MemOwned = false; // This is why I fucking hate this language

          return *this;
      }
      
      // Copy assignment operator
      Buffer& operator=(const Buffer& other) {
        if (m_MemOwned && m_Buffer != NULL) {
          switch(m_Device) {
            case Device::CPU: {
              delete[] m_Buffer;
              break;
            }
            case Device::GPU: {
              cudaFree(m_Buffer);
              break;
            }
          }
        }
         
        m_Size = other.m_Size;
        m_Buffer = other.m_Buffer;  
        m_Device = other.m_Device;
        m_MemOwned = false;

        return *this;
      }

      Buffer() {};

      ~Buffer() {
          if (m_MemOwned) {
            if (m_Buffer != NULL) {
              delete m_Buffer;
            }
          }
      }

      T operator[](size_t idx) { 
        if (m_Device != Device::CPU) LOG_ERROR("Cannot access data of a GPU buffer!");

        return m_Buffer[idx];
      }
};

}

#endif // MEMORY_H
