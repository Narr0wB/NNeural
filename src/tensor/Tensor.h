
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
        T* m_Data;

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
            if (!(typeid(T) == typeid(FLOAT32) || typeid(T) == typeid(FLOAT64) || typeid(T) == typeid(INT32) || typeid(T) == typeid(INT64))) {
                LOG_ERROR("[ERROR] (Tensor::Tensor) Unsupported tensor type!");
            }
            
            if (shape.size() > 3) {
                LOG_ERROR("[ERROR] (Tensor::Tensor) The maximum supported rank is 3!");
            }

            for (uint32_t dimentions : shape) {
                m_Size *= dimentions;
            }

            m_Data = new T[m_Size];
        }

        Tensor(T val) {
            if (!(typeid(T) == typeid(FLOAT32) || typeid(T) == typeid(FLOAT64) || typeid(T) == typeid(INT32) || typeid(T) == typeid(INT64))) {
                LOG_ERROR("[ERROR] (Tensor::Tensor) Unsupported tensor type!");
            }

            m_Shape = {1};
            m_Size = 1;

            m_Data = new T[m_Size];

            m_Data[0] = val;
        }

        Tensor(uint32_t x, uint32_t y = 0, uint32_t z = 0) {
            if (!(typeid(T) == typeid(FLOAT32) || typeid(T) == typeid(FLOAT64) || typeid(T) == typeid(INT32) || typeid(T) == typeid(INT64))) {
                LOG_ERROR("[ERROR] (Tensor::Tensor) Unsupported tensor type!");
            }

            if (x == 0) {
                LOG_ERROR("[ERROR] (Tensor::Tensor) Invalid initialization index!");
            }

            if (y == 0 && z == 0) {
                m_Shape = TensorShape{x};
                m_Size = x;

                m_Data = new T[m_Size];

                return;
            }

            if (z == 0) {
                m_Shape = TensorShape{x, y};
                m_Size = x*y;

                m_Data = new T[m_Size];

                return;
            }

            m_Shape = TensorShape{x, y, z};
            m_Size = x*y*z;

            m_Data = new T[m_Size];
        }

        Tensor(std::initializer_list<T> list) {
            if (!(typeid(T) == typeid(FLOAT32) || typeid(T) == typeid(FLOAT64) || typeid(T) == typeid(INT32) || typeid(T) == typeid(INT64))) {
                LOG_ERROR("[ERROR] (Tensor::Tensor) Unsupported tensor type!");
            }

            m_Shape = TensorShape{list.size()};
            m_Size = list.size();

            m_Data = new T[m_Size];

            uint32_t i;
            for (auto element : list) {
                m_Data[getIndex(i++)] = element;
            }
        }

        Tensor(std::initializer_list<std::initializer_list<T>> list) {
            if (!(typeid(T) == typeid(FLOAT32) || typeid(T) == typeid(FLOAT64) || typeid(T) == typeid(INT32) || typeid(T) == typeid(INT64))) {
                LOG_ERROR("[ERROR] (Tensor::Tensor) Unsupported tensor type!");
            }

            auto first = list.begin();
            m_Shape = TensorShape{first->size(), list.size()};
            m_Size = list.size() * first->size();


            m_Data = new T[m_Size];

            uint32_t i = 0;
            for (auto rank_1_tensor : list) {
                if (rank_1_tensor.size() != m_Shape[0]) {
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
            if (!(typeid(T) == typeid(FLOAT32) || typeid(T) == typeid(FLOAT64) || typeid(T) == typeid(INT32) || typeid(T) == typeid(INT64))) {
                LOG_ERROR("[ERROR] (Tensor::Tensor) Unsupported tensor type!");
            }

            auto first = list.begin();
            auto second = first->begin();
            m_Shape = TensorShape{list.size(), first->size(), second->size()};
            m_Size = list.size() * first->size() * second->size();

            m_Data = new T[m_Size];
            
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

                    std::memcpy(_rank_1_tensor.m_Data, m_Data + getIndex(index), m_Shape[1] * sizeof(T));

                    return _rank_1_tensor;
                }
                
                case 3: {
                    Tensor<T> _rank_2_tensor(m_Shape[1], m_Shape[2]);

                    std::memcpy(_rank_2_tensor.m_Data, m_Data + getIndex(index), m_Shape[1] * m_Shape[2] * sizeof(T));

                    return _rank_2_tensor;
                }
                
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

        void set(Tensor<T>, uint32_t x, uint32_t y = 0) [

        ]

        inline size_t size() { return m_Size; };
        inline size_t rank() { return m_Shape.size(); }
        inline TensorShape shape() { returm m_Shape; }

        // TENSOR OPERATIONS --------------------------------------------------------------------------------------------------------------


        
};

template <typename T>
Tensor<T> tensormul(Tensor<T> a, Tensor<T> b) {
    uint32_t a_3_rank = a.size() == 3 ? a.shape[0] : 1;
    uint32_t b_3_rank = b.size() == 3 ? b.shape[0] : 1;

    uint32_t a_2_rank = a.size() == 1 ? 1 : a.shape[a.size() - 2];
    uint32_t b_2_rank = b.size() == 1 ? 1 : b.shape[b.size() - 2];

    uint32_t a_1_rank = a.shape[a.size() - 1];
    uint32_t b_1_rank = b.shape[b.size() - 1];

    uint32_t result_3_rank = std::max(a_3_rank, b_3_rank);

    if (a_3_rank != b_3_rank and (a_3_rank != 1 or b_3_rank != 1)) {
        LOG_ERROR("[ERROR] (tensormul) Invalid input tensors!");
    }

    Tensor<T> _result = result_3_rank != 1 ? Tensor(result_3_rank, a_2_rank, b_1_rank) : Tensor(a_2_rank, b_1_rank);

    for (int i = 0; i < result_3_rank; ++i) {

    }


    return Tensor<T>();
}

#endif // TENSOR_H