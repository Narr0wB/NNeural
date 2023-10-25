
#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <iostream>
#include <memory>

#include "Operations.h"
#include "../types.h"
#include "../utils/Log.h"

typedef std::vector<uint32_t> TensorShape;
#define MAX_RANK 3

template <typename T>
class Tensor {
    private:
        TensorShape m_Shape;
        size_t m_Size = 1;
        std::unique_ptr<T[]> m_Data;

        size_t getIndex(uint32_t x, uint32_t y = 0, uint32_t z = 0) {
            if (m_Shape.size() == 1) {
                return x;
            }

            if (m_Shape.size() == 2) {
                return (x * m_Shape[1]) + y;
            }

            return (x * m_Shape[1] * m_Shape[2]) + (y * m_Shape[2]) + z;
        }

    public:
        Tensor() {
            m_Shape = TensorShape{0};
            m_Size = 0;
        }

        Tensor(TensorShape shape) {
            if (!(typeid(T) == typeid(FP32) || typeid(T) == typeid(FP64) || typeid(T) == typeid(I32) || typeid(T) == typeid(I64))) {
                LOG_ERROR("[ERROR] (Tensor::Tensor) Unsupported tensor type!");
            }
            
            if (shape.size() > 3) {
                LOG_ERROR("[ERROR] (Tensor::Tensor) The maximum supported rank is 3!");
            }

            for (uint32_t dimentions : shape) {
                m_Size *= dimentions;
            }

            m_Data = std::make_unique<T[]>(m_Size);;
        }

        Tensor(T val) {
            if (!(typeid(T) == typeid(FP32) || typeid(T) == typeid(FP64) || typeid(T) == typeid(I32) || typeid(T) == typeid(I64))) {
                LOG_ERROR("[ERROR] (Tensor::Tensor) Unsupported tensor type!");
            }

            m_Shape = {1};
            m_Size = 1;

            m_Data = std::make_unique<T[]>(m_Size);

            m_Data[0] = val;
        }

        Tensor(uint32_t x, uint32_t y = 0, uint32_t z = 0) {
            if (!(typeid(T) == typeid(FP32) || typeid(T) == typeid(FP64) || typeid(T) == typeid(I32) || typeid(T) == typeid(I64))) {
                LOG_ERROR("[ERROR] (Tensor::Tensor) Unsupported tensor type!");
            }

            if (x == 0) {
                LOG_ERROR("[ERROR] (Tensor::Tensor) Invalid initialization index!");
            }

            if (y == 0 && z == 0) {
                m_Shape = TensorShape{x};
                m_Size = x;

                m_Data = std::make_unique<T[]>(m_Size);

                return;
            }

            if (z == 0) {
                m_Shape = TensorShape{x, y};
                m_Size = x*y;

                m_Data = std::make_unique<T[]>(m_Size);

                return;
            }

            m_Shape = TensorShape{x, y, z};
            m_Size = x*y*z;

            m_Data = std::make_unique<T[]>(m_Size);
        }

        Tensor(std::initializer_list<T> list) {
            if (!(typeid(T) == typeid(FP32) || typeid(T) == typeid(FP64) || typeid(T) == typeid(I32) || typeid(T) == typeid(I64))) {
                LOG_ERROR("[ERROR] (Tensor::Tensor) Unsupported tensor type!");
            }

            m_Shape = TensorShape{list.size()};
            m_Size = list.size();

            m_Data = std::make_unique<T[]>(m_Size);

            uint32_t i = 0;
            for (auto element : list) {
                m_Data[getIndex(i++)] = element;
            }
        }

        Tensor(std::initializer_list<std::initializer_list<T>> list) {
            if (!(typeid(T) == typeid(FP32) || typeid(T) == typeid(FP64) || typeid(T) == typeid(I32) || typeid(T) == typeid(I64))) {
                LOG_ERROR("[ERROR] (Tensor::Tensor) Unsupported tensor type!");
            }

            auto first = list.begin();
            m_Shape = TensorShape{list.size(), first->size()};
            m_Size = list.size() * first->size();


            m_Data = std::make_unique<T[]>(m_Size);

            uint32_t i = 0;
            for (auto rank_1_tensor : list) {
                if (rank_1_tensor.size() != m_Shape[1]) {
                    LOG_ERROR("[ERROR] (Tensor::Tensor) Tensor sizes dont match!");
                }

                uint32_t j = 0;
                for (auto element : rank_1_tensor) {
                    m_Data[getIndex(i, j++)] = element;
                }

                i++;
            }
        } 

        Tensor(std::initializer_list<std::initializer_list<std::initializer_list<T>>> list) {
            if (!(typeid(T) == typeid(FP32) || typeid(T) == typeid(FP64) || typeid(T) == typeid(I32) || typeid(T) == typeid(I64))) {
                LOG_ERROR("[ERROR] (Tensor::Tensor) Unsupported tensor type!");
            }

            auto first = list.begin();
            auto second = first->begin();
            m_Shape = TensorShape{list.size(), first->size(), second->size()};
            m_Size = list.size() * first->size() * second->size();

            m_Data = std::make_unique<T[]>(m_Size);
            
            uint32_t i = 0;
            for (auto rank_2_tensor : list) {
                if (rank_2_tensor.size() != m_Shape[1]) {
                    LOG_ERROR("[ERROR] (Tensor::Tensor) Tensor sizes dont match!");
                }

                uint32_t j = 0;
                for (auto rank_1_tensor : rank_2_tensor) {
                    if (rank_1_tensor.size() != m_Shape[2]) {
                        LOG_ERROR("[ERROR] (Tensor::Tensor) Tensor sizes dont match!");
                    }
                    
                    uint32_t k = 0;
                    for (auto element : rank_1_tensor) {
                        m_Data[getIndex(i, j, k++)] = element; 
                    }
                    
                    j++;
                }

                i++;
            }
        }

        Tensor<T> broadcast(TensorShape broadcast_shape) {
            size_t broadcast_rank = broadcast_shape.size();
            size_t current_rank = m_Size.size();

            if (broadcast_rank < current_rank) {
                LOG_ERROR("[ERROR] (Tensor::broadcast) Invalid broadcast shape!");
            }

            m_Shape.insert(0, broadcast_rank - current_rank, 1);

            for (size_t i = 0; i < MAX_RANK; ++i) {
                m_Shape[i] = std::max(m_Shape[i], broadcast_shape[i]);
            }

            realloc();

            


        }

        operator T() { 
            if (m_Shape.size() != 1 && m_Shape[0] != 1) { 
                LOG_ERROR("[ERROR] Only scalar tensors can be converted to numbers!"); 
            } 

            return m_Data[0];
        }

        Tensor<T> operator[] (uint32_t index) {
            if (index > m_Shape[0]) {
                LOG_ERROR("[ERROR] (Tensor::operator[]) Invalid index!");
            }

            switch (m_Shape.size()) {
                case 1: {
                    return Tensor<T>(m_Data[getIndex(index)]);
                }
                
                case 2: {
                    Tensor<T> _rank_1_tensor(m_Shape[1]);

                    std::memcpy(_rank_1_tensor.m_Data.get(), m_Data.get() + getIndex(index), m_Shape[1] * sizeof(T));

                    return _rank_1_tensor;
                }
                
                case 3: {
                    Tensor<T> _rank_2_tensor(m_Shape[1], m_Shape[2]);

                    std::memcpy(_rank_2_tensor.m_Data.get(), m_Data.get() + getIndex(index), m_Shape[1] * m_Shape[2] * sizeof(T));

                    return _rank_2_tensor;
                }
                
                return Tensor<T>();
            }
        }

        T operator() (uint32_t x, uint32_t y = 0, uint32_t z = 0) {
            if (x > m_Shape[0]) {
                LOG_ERROR("[ERROR] (Tensor::operator()) Invalid access x index!");
            }

            if (y > m_Shape[1]) {
                LOG_ERROR("[ERROR] (Tensor::operator()) Invalid access y index!");
            }

            if (z > m_Shape[2]) {
                LOG_ERROR("[ERROR] (Tensor::operator()) Invalid access z index!");
            }

            return m_Data[getIndex(x, y, z)];
        }
        
        void set(T, uint32_t x, uint32_t y = 0, uint32_t z = 0) {

        }

        // void set(Tensor<T>, uint32_t x, uint32_t y = 0) [

        // ]

        inline size_t size() { return m_Size; };
        inline size_t rank() { return m_Shape.size(); }
        inline TensorShape shape() { return m_Shape; }
        inline T* data() { return m_Data.get(); }

        // TENSOR OPERATIONS --------------------------------------------------------------------------------------------------------------

};

template <typename T>
Tensor<T> tensormul(Tensor<T>& a, Tensor<T>& b) {

    if (a.rank() > 2 || b.rank() > 2) {
        uint32_t a_3_rank = a.rank() == 3 ? a.shape()[0] : 1;
        uint32_t b_3_rank = b.rank() == 3 ? b.shape()[0] : 1;

        uint32_t a_2_rank = a.rank() == 1 ? 1 : a.shape()[a.rank() - 2];
        uint32_t b_2_rank = b.rank() == 1 ? 1 : b.shape()[b.rank() - 2];

        uint32_t a_1_rank = a.shape()[a.rank() - 1];
        uint32_t b_1_rank = b.shape()[b.rank() - 1];

        if (a_3_rank != b_3_rank and (a_3_rank != 1 and b_3_rank != 1)) {
            LOG_ERROR("[ERROR] (tensormul) Invalid input tensors!");
        }

        uint32_t result_3_rank = std::max(a_3_rank, b_3_rank);
        uint32_t result_2_rank = a_2_rank;
        uint32_t result_1_rank = b_1_rank;

        Tensor<T> _result = result_3_rank != 1 ? Tensor<T>(result_3_rank, result_2_rank, result_1_rank) : Tensor<T>(result_2_rank, result_1_rank);
        T* result_data = _result.data();

        MatrixShape mat_1_shape = {a_2_rank, a_1_rank};
        MatrixShape mat_2_shape = {b_2_rank, b_1_rank};

        for (uint32_t i = 0; i < result_3_rank; ++i) {
            T* _result_data = result_data + i * (result_2_rank * result_1_rank);

            // If Tensor A's 3rd rank is greater than 1, then the internal matrix will change
            T* mat_1_data = a.data() + (a_3_rank != 1 ? i : 0) * (a_2_rank * a_1_rank);

            // If Tensor B's 3rd rank is greater than 1, then the internal matrix will change
            T* mat_2_data = b.data() + (b_3_rank != 1 ? i : 0) * (b_2_rank * b_1_rank);

            // Now calculate the single matrix multiplication for each internal matrix
            matmul(TYPE_TO_ENUM(T), (void*)_result_data, (void*)mat_1_data, (void*)mat_2_data, mat_1_shape, mat_2_shape);
        }

        return _result;
    }
    else {

        uint32_t a_2_rank = a.rank() == 1 ? a.shape()[a.rank() - 1] : a.shape()[a.rank() - 2];
        uint32_t b_2_rank = b.rank() == 1 ? b.shape()[b.rank() - 1] : b.shape()[b.rank() - 2];

        uint32_t a_1_rank = a.rank() == 1 ? 1 : a.shape()[a.rank() - 1];
        uint32_t b_1_rank = b.rank() == 1 ? 1 : b.shape()[b.rank() - 1];

        if (a_1_rank != b_2_rank) {
            LOG_ERROR("[ERROR] (tensormul) Invalid input tensors!");
        }

        uint32_t result_2_rank = a_2_rank;
        uint32_t result_1_rank = b_1_rank;

        Tensor<T> _result = Tensor<T>(result_2_rank, result_1_rank);
        T* result_data = _result.data();

        // if we have a one-dimentional Tensor, we treat it as if it were a column vector
        MatrixShape mat_1_shape = {a_2_rank, a_1_rank};
        MatrixShape mat_2_shape = {b_2_rank, b_1_rank};

        T* mat_1_data = a.data();
        T* mat_2_data = b.data();

        matmul(TYPE_TO_ENUM(T), (void*)result_data, (void*)mat_1_data, (void*)mat_2_data, mat_1_shape, mat_2_shape);

        return _result;
    }

    return Tensor<T>();
}

#endif // TENSOR_H