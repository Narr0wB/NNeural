
#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <iostream>
#include <memory>
#include <functional>
#include <numeric>

#include "Operations.h"
#include "../types.h"
#include "../utils/Log.h"

typedef std::vector<uint32_t> TensorShape;
#define MAX_RANK 3

int func(int eddu);

template <typename T>
class BroadcastTensor {
    private:
        TensorShape m_Shape;
        TensorShape m_OriginalShape;

        size_t m_Size = 1;
        std::shared_ptr<T[]> m_Data; 

        // Get data's index in the internal buffer
        size_t _index(std::initializer_list<uint32_t> parameters) {
            size_t index = 0;
            if (parameters.size() != m_OriginalShape.size()) {
                //LOG_ERROR("DEBUG: EROR");
            }

            for (int i = m_OriginalShape.size() - 1; i > -1; --i) {
                if (m_OriginalShape[i] != 1) {
                    index += *(parameters.begin() + i) * std::accumulate(m_OriginalShape.begin() + i + 1, m_OriginalShape.end(), 1, std::multiplies<T>());
                }
            }

            return index;
        }

    public:

        BroadcastTensor(std::shared_ptr<T[]>& original_data, TensorShape& shape, TensorShape& original_shape) : m_Data(original_data), m_Shape(shape), m_OriginalShape(original_shape) {}

        T operator() (uint32_t x, uint32_t y, uint32_t z) {
            return m_Data[_index({x, y, z})];
        }

        inline size_t rank() { return m_Shape.size(); }

        template <typename B>
        friend std::ostream& operator<<(std::ostream& out, BroadcastTensor<B>& tensor) {
            switch (tensor.rank()) {
                case 1: {
                    out << "{";

                    for (uint32_t i = 0; i < tensor.m_Shape[0] - 1; ++i) {
                        out << tensor.m_Data[tensor._index({i})] << ", ";
                    }

                    out << tensor.m_Data[tensor._index({tensor.m_Shape[0] - 1})] << "}" << std::endl;
                    break;
                }

                case 2: {
                    out << "{" << std::endl;
                    for (uint32_t j = 0; j < tensor.m_Shape[0]; ++j) {
                        out << "    ";
                        out << "{";

                        for (uint32_t i = 0; i < tensor.m_Shape[1] - 1; ++i) {
                            out << tensor.m_Data[tensor._index({j, i})] << ", ";
                        }

                        out << tensor.m_Data[tensor._index({j, tensor.m_Shape[1] - 1})] << "}";
                        out << std::endl;
                    }
                    out << "}" << std::endl;
                    break;
                }

                case 3: {
                    out << "{" << std::endl;

                    for (uint32_t k = 0; k < tensor.m_Shape[0]; ++k) {

                        out << "    ";
                        out << "{" << std::endl;
                        out << "    ";

                        for (uint32_t j = 0; j < tensor.m_Shape[1]; ++j) {
                            out << "    ";
                            out << "{";

                            for (uint32_t i = 0; i < tensor.m_Shape[2] - 1; ++i) {
                                out << tensor.m_Data[tensor._index({k, j, i})] << ", ";
                            }

                            out << tensor.m_Data[tensor._index({k, j, tensor.m_Shape[2] - 1})] << "}";
                            out << std::endl;
                            out << "    ";
                        }

                        out << "}" << std::endl;
                    }

                    out << "}" << std::endl;
                    break;
                }
            }

            return out;
        }
};


template <typename T>
class Tensor {
    private:
        TensorShape m_Shape;
        size_t m_Size = 1;
        std::shared_ptr<T[]> m_Data;

        // Get data's index in the internal buffer
        size_t _index(std::initializer_list<uint32_t> parameters) {
            size_t index = 0;
            if (parameters.size() != m_Shape.size()) {
                //LOG_ERROR("DEBUG: EROR");
            }

            for (int i = m_Shape.size() - 1; i > -1; --i) {
                index += *(parameters.begin() + i) * std::accumulate(m_Shape.begin() + i + 1, m_Shape.end(), 1, std::multiplies<T>());
            }

            return index;
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
                m_Data[_index({i++})] = element;
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
                    m_Data[_index({i, j++})] = element;
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
                        m_Data[_index({i, j, k++})] = element; 
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
                    return Tensor<T>(m_Data[_index({index})]);
                }
                
                case 2: {
                    Tensor<T> _rank_1_tensor(m_Shape[1]);

                    std::memcpy(_rank_1_tensor.m_Data.get(), m_Data.get() + _index({index}), m_Shape[1] * sizeof(T));

                    return _rank_1_tensor;
                }
                
                case 3: {
                    Tensor<T> _rank_2_tensor(m_Shape[1], m_Shape[2]);

                    std::memcpy(_rank_2_tensor.m_Data.get(), m_Data.get() + _index({index}), m_Shape[1] * m_Shape[2] * sizeof(T));

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

            return m_Data[_index({x, y, z})];
        }
        
        void set(T val, uint32_t x, uint32_t y, uint32_t z) {
            m_Data[_index({x, y, z})] = val;
        }

        BroadcastTensor<T> broadcast(TensorShape broadcast_shape) {
            size_t broadcast_rank = broadcast_shape.size();
            size_t current_rank = rank();
            TensorShape original_shape = m_Shape;

            if (broadcast_rank < current_rank) {
                LOG_ERROR("[ERROR] (Tensor::broadcast) Invalid broadcast shape!");
            }

            original_shape.insert(original_shape.begin(), broadcast_rank - current_rank, 1);
            TensorShape new_shape = original_shape;

            for (size_t i = 0; i < MAX_RANK; ++i) {
                if (new_shape[i] != broadcast_shape[i] && new_shape[i] != 1){
                    LOG_ERROR("[ERROR] (Tensor::broadcast) Cannot broadcast a ({}, {}, {}) shape to a ({}, {}, {}) shape", rank() == 3 ? original_shape[0] : 1, rank() >= 2 ? original_shape[rank() - 2] : 1, original_shape[rank() - 1], broadcast_shape.size() == 3 ? broadcast_shape[0] : 1, broadcast_shape.size() >= 2 ? broadcast_shape[broadcast_shape.size() - 2] : 1, broadcast_shape[broadcast_shape.size() - 1]);
                }

                new_shape[i] = std::max(new_shape[i], broadcast_shape[i]);
            }

            return BroadcastTensor<T>(m_Data, new_shape, original_shape);
        }

        // void set(Tensor<T>, uint32_t x, uint32_t y = 0) [

        // ]

        inline size_t size() { return m_Size; };
        inline size_t rank() { return m_Shape.size(); }
        inline TensorShape shape() { return m_Shape; }
        inline T* data() { return m_Data.get(); }

        // TENSOR OPERATIONS --------------------------------------------------------------------------------------------------------------
        
        template <typename B>
        friend std::ostream& operator<<(std::ostream& out, Tensor<B>& tensor) {
            switch (tensor.rank()) {
                case 1: {
                    out << "{";

                    for (uint32_t i = 0; i < tensor.m_Shape[0] - 1; ++i) {
                        out << tensor.m_Data[tensor._index({i})] << ", ";
                    }

                    out << tensor.m_Data[tensor._index({tensor.m_Shape[0] - 1})] << "}" << std::endl;
                    break;
                }

                case 2: {
                    out << "{" << std::endl;
                    for (uint32_t j = 0; j < tensor.m_Shape[0]; ++j) {
                        out << "    ";
                        out << "{";

                        for (uint32_t i = 0; i < tensor.m_Shape[1] - 1; ++i) {
                            out << tensor.m_Data[tensor._index({j, i})] << ", ";
                        }

                        out << tensor.m_Data[tensor._index({j, tensor.m_Shape[1] - 1})] << "}";
                        out << std::endl;
                    }
                    out << "}" << std::endl;
                    break;
                }

                case 3: {
                    out << "{" << std::endl;

                    for (uint32_t k = 0; k < tensor.m_Shape[0]; ++k) {

                        out << "    ";
                        out << "{" << std::endl;
                        out << "    ";

                        for (uint32_t j = 0; j < tensor.m_Shape[1]; ++j) {
                            out << "    ";
                            out << "{";

                            for (uint32_t i = 0; i < tensor.m_Shape[2] - 1; ++i) {
                                out << tensor.m_Data[tensor._index({k, j, i})] << ", ";
                            }

                            out << tensor.m_Data[tensor._index({k, j, tensor.m_Shape[2] - 1})] << "}";
                            out << std::endl;
                            out << "    ";
                        }

                        out << "}" << std::endl;
                    }

                    out << "}" << std::endl;
                    break;
                }
            }

            return out;
        }
};

template <typename T>
Tensor<T> tensormul(Tensor<T>& a, Tensor<T>& b) {
    uint32_t a_3_rank = a.rank() == 3 ? a.shape()[0] : 1;
    uint32_t b_3_rank = b.rank() == 3 ? b.shape()[0] : 1;

    uint32_t a_2_rank = a.rank() == 1 ? 1 : a.shape()[a.rank() - 2];
    uint32_t b_2_rank = b.rank() == 1 ? 1 : b.shape()[b.rank() - 2];

    uint32_t a_1_rank = a.shape()[a.rank() - 1];
    uint32_t b_1_rank = b.shape()[b.rank() - 1];

    if (a.rank() > b.rank()) {
        BroadcastTensor<T> b_broadcast = b.broadcast({a_3_rank, a_1_rank, b_1_rank});
        Tensor<T> result(a_3_rank, a_2_rank, b_1_rank);

        for (uint32_t r = 0; r < a_3_rank; ++r) {

            for (uint32_t i = 0; i < a_2_rank; ++i) {
                for (uint32_t j = 0; j < b_1_rank; ++j) {
                    T temp_sum = 0;

                    for (uint32_t k = 0; k < a_1_rank; ++k) {
                        temp_sum += (a(r, i, k) * b_broadcast(r, k, j));
                    }

                    result.set(temp_sum, r, i, j);
                }

                std::cout << result << std::endl;
            }

        }

        return result;
    }


    // if (a.rank() > 2 || b.rank() > 2) {
        

    //     if (a_3_rank != b_3_rank and (a_3_rank != 1 and b_3_rank != 1)) {
    //         LOG_ERROR("[ERROR] (tensormul) Invalid input tensors!");
    //     }

    //     uint32_t result_3_rank = std::max(a_3_rank, b_3_rank);
    //     uint32_t result_2_rank = a_2_rank;
    //     uint32_t result_1_rank = b_1_rank;

    //     Tensor<T> _result = result_3_rank != 1 ? Tensor<T>(result_3_rank, result_2_rank, result_1_rank) : Tensor<T>(result_2_rank, result_1_rank);
    //     T* result_data = _result.data();

    //     MatrixShape mat_1_shape = {a_2_rank, a_1_rank};
    //     MatrixShape mat_2_shape = {b_2_rank, b_1_rank};

    //     for (uint32_t i = 0; i < result_3_rank; ++i) {
    //         T* _result_data = result_data + i * (result_2_rank * result_1_rank);

    //         // If Tensor A's 3rd rank is greater than 1, then the internal matrix will change
    //         T* mat_1_data = a.data() + (a_3_rank != 1 ? i : 0) * (a_2_rank * a_1_rank);

    //         // If Tensor B's 3rd rank is greater than 1, then the internal matrix will change
    //         T* mat_2_data = b.data() + (b_3_rank != 1 ? i : 0) * (b_2_rank * b_1_rank);

    //         // Now calculate the single matrix multiplication for each internal matrix
    //         //matmul(TYPE_TO_ENUM(T), (void*)_result_data, (void*)mat_1_data, (void*)mat_2_data, mat_1_shape, mat_2_shape);


    //     }

    //     return _result;
    // }
    // else {

    //     uint32_t a_2_rank = a.rank() == 1 ? a.shape()[a.rank() - 1] : a.shape()[a.rank() - 2];
    //     uint32_t b_2_rank = b.rank() == 1 ? b.shape()[b.rank() - 1] : b.shape()[b.rank() - 2];

    //     uint32_t a_1_rank = a.rank() == 1 ? 1 : a.shape()[a.rank() - 1];
    //     uint32_t b_1_rank = b.rank() == 1 ? 1 : b.shape()[b.rank() - 1];

    //     if (a_1_rank != b_2_rank) {
    //         LOG_ERROR("[ERROR] (tensormul) Invalid input tensors!");
    //     }

    //     uint32_t result_2_rank = a_2_rank;
    //     uint32_t result_1_rank = b_1_rank;

    //     Tensor<T> _result = Tensor<T>(result_2_rank, result_1_rank);
    //     T* result_data = _result.data();

    //     // if we have a one-dimentional Tensor, we treat it as if it were a column vector
    //     MatrixShape mat_1_shape = {a_2_rank, a_1_rank};
    //     MatrixShape mat_2_shape = {b_2_rank, b_1_rank};

    //     T* mat_1_data = a.data();
    //     T* mat_2_data = b.data();

    //     matmul(TYPE_TO_ENUM(T), (void*)result_data, (void*)mat_1_data, (void*)mat_2_data, mat_1_shape, mat_2_shape);

    //     return _result;
    // }

    return Tensor<T>();
}

#endif // TENSOR_H