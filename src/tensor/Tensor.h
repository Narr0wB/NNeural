
#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <iostream>
#include <memory>
#include "../types.h"
#include "../utils/Log.h"

typedef std::vector<uint32_t> TensorShape;

template <typename T>
class Tensor {
    private:
        TensorShape m_Shape;
        size_t m_Size = 1;
        std::unique_ptr<T> m_Data;

    public:
        Tensor(TensorShape shape) {
            if (shape.size() > 3) {
                LOG_ERROR("[ERROR] The maximum supported rank is 3! Aborting...");
            }

            if (!(typeid(T) == typeid(FLOAT32) || typeid(T) == typeid(FLOAT64) || typeid(T) == typeid(INT32) || typeid(T) == typeid(INT64))) {
                LOG_ERROR("[ERROR] Unsupported tensor type! Aborting...");
            }

            for (uint32_t dimentions : shape) {
                m_Size *= dimentions;
            }

            m_Data = std::unique_ptr<T>(new T[m_Size]);
        }

        inline size_t getSize() { return m_Size; };
};

#endif // TENSOR_H