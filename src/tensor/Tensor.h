
#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <iostream>
#include <functional>
#include <numeric>

#include "../types.h"
#include "../utils/Log.h"
#include "../utils/Hardware.h"
#include "../utils/Memory.h"

#define MAX_RANK 3

typedef std::vector<size_t> TensorShape;
std::ostream& operator<<(std::ostream& out, TensorShape shape);

// template <typename T>
// class BroadcastTensor {
//     private:
//         TensorShape m_Shape;
//         TensorShape m_OriginalShape;
//
//         size_t m_Size;
//         size_t m_OriginalSize;
//         std::shared_ptr<T[]> m_Data; 
//
//         // Get data's index in the internal buffer
//         size_t _index(std::initializer_list<uint32_t> parameters) const {
//             size_t index = 0;
//             if (parameters.size() != m_OriginalShape.size()) {
//                 //LOG_ERROR("DEBUG: EROR");
//             }
//
//             for (int i = 0; i < m_OriginalShape.size(); ++i) {
//                 if (m_OriginalShape[rank() - i - 1] != 1) {
//                     auto current_parameter = *(parameters.end() - i - 1);
//                 
//                     index += current_parameter * std::accumulate(m_OriginalShape.end() - i, m_OriginalShape.end(), 1, std::multiplies<T>());
//                 }
//             }
//
//             return index;
//         }
//
//     public:
//
//         BroadcastTensor(std::shared_ptr<T[]>& original_data, size_t size, size_t original_size, TensorShape& shape, TensorShape& original_shape) : m_Data(original_data), m_Shape(shape), m_OriginalShape(original_shape), m_Size(size), m_OriginalSize(original_size) {}
//
//         T operator() (uint32_t x, uint32_t y, uint32_t z) {
//             return m_Data[_index({x, y, z})];
//         }
//
//         T get(uint32_t internal_index) {
//             if (internal_index > m_Size - 1) {
//                 LOG_ERROR("[ERROR] (Tensor::get) Invalid internal access index!");
//             }
//
//             return m_Data[internal_index % m_OriginalSize];
//         }
//
//         inline size_t rank() { return m_Shape.size(); }
//         inline FP64 identifier() { return ((FP64)m_Size / (m_Size + 3)) * rank(); }
//
//         template <typename B>
//         friend std::ostream& operator<<(std::ostream& out, BroadcastTensor<B>& tensor) {
//             switch (tensor.rank()) {
//                 case 1: {
//                     out << "{";
//
//                     for (uint32_t i = 0; i < tensor.m_Shape[0] - 1; ++i) {
//                         out << tensor.m_Data[tensor._index({i})] << ", ";
//                     }
//
//                     out << tensor.m_Data[tensor._index({tensor.m_Shape[0] - 1})];
//
//                     out << "}";
//                     break;
//                 }
//
//                 case 2: {
//                     out << "{" << std::endl;
//                     for (uint32_t j = 0; j < tensor.m_Shape[0]; ++j) {
//                         out << "    ";
//                         out << "{";
//
//                         for (uint32_t i = 0; i < tensor.m_Shape[1] - 1; ++i) {
//                             out << tensor.m_Data[tensor._index({j, i})] << ", ";
//                         }
//
//                         out << tensor.m_Data[tensor._index({j, tensor.m_Shape[1] - 1})] << "}";
//                         out << std::endl;
//                     }
//                     out << "}";
//                     break;
//                 }
//
//                 case 3: {
//                     out << "{" << std::endl;
//
//                     for (uint32_t k = 0; k < tensor.m_Shape[0]; ++k) {
//
//                         out << "    ";
//                         out << "{" << std::endl;
//                         out << "    ";
//
//                         for (uint32_t j = 0; j < tensor.m_Shape[1]; ++j) {
//                             out << "    ";
//                             out << "{";
//
//                             for (uint32_t i = 0; i < tensor.m_Shape[2] - 1; ++i) {
//                                 out << tensor.m_Data[tensor._index({k, j, i})] << ", ";
//                             }
//
//                             out << tensor.m_Data[tensor._index({k, j, tensor.m_Shape[2] - 1})] << "}";
//                             out << std::endl;
//                             out << "    ";
//                         }
//
//                         out << "}" << std::endl;
//                     }
//
//                     out << "}";
//                     break;
//                 }
//             }
//
//             return out;
//         }
// };

template <typename T>
class Tensor {
    private:
        TensorShape m_Shape;
        TensorShape m_Strides;

        size_t m_Size;
        Memory::Buffer<T> m_Data;

        // Get data's index in the internal buffer
        size_t _index(std::initializer_list<size_t> parameters) const {
            //LOG_INFO("{} {}", 1, 1);

            size_t index = 0;
            if (parameters.size() != m_Shape.size()) {
                LOG_ERROR("Invalid access parameters");
            }

            //TensorShape tmp{3, 1};

            for (int i = 0; i < parameters.size(); ++i) {
                index += parameters.begin()[i] * m_Strides[i];
            }

            // LOG_INFO("{} {}, {} {}, {} {}", m_Strides[0], parameters.begin()[0], m_Strides[1], parameters.begin()[1], m_Strides[2], parameters.begin()[2]);
            // LOG_INFO("{}", index);

            return index;
        }

        void _calculate_index_offsets(const TensorShape& shape) {
            for (int i = 1; i < shape.size(); ++i) {
                // Calculate the offets for a particular dimention in the Memory buffer
                if (shape[i] == 1) {
                    m_Strides.push_back(0);
                    continue;
                }

                m_Strides.push_back(std::accumulate(shape.begin() + i, shape.end(), 1, std::multiplies<T>()));
            }
            m_Strides.push_back(1);
        }

        Tensor(Memory::Buffer<T> buf, TensorShape tsh) : 
        m_Data(buf),
        m_Shape(tsh),
        m_Size(1) {
            for (uint32_t dimention : m_Shape) {
                m_Size *= dimention;
            }

            _calculate_index_offsets(m_Shape);
        }

        Tensor(Memory::Buffer<T> buf, TensorShape tsh, TensorShape original_shape) : 
        m_Data(buf),
        m_Shape(tsh),
        m_Size(1) {
            for (uint32_t dimention : m_Shape) {
                m_Size *= dimention;
            }

            _calculate_index_offsets(original_shape);
        }

    public:
        Tensor() {
            m_Shape = TensorShape{0};
            m_Strides = TensorShape{0};
            m_Size = 0;
        }

        Tensor(TensorShape shape) : m_Shape(shape), m_Size(1) {
            if (!(typeid(T) == typeid(FP32) || typeid(T) == typeid(FP64) || typeid(T) == typeid(I32) || typeid(T) == typeid(I64))) {
                LOG_ERROR("[ERROR] (Tensor::Tensor) Unsupported tensor type!");
            }

            for (uint32_t dimention : m_Shape) {
                m_Size *= dimention;
            }

            _calculate_index_offsets(shape);

            m_Data = Memory::Buffer<T>(m_Size);
        }

        Tensor(T val) {
            if (!(typeid(T) == typeid(FP32) || typeid(T) == typeid(FP64) || typeid(T) == typeid(I32) || typeid(T) == typeid(I64))) {
                LOG_ERROR("[ERROR] (Tensor::Tensor) Unsupported tensor type!");
            }

            m_Shape = {1};
            m_Strides = {1};
            m_Size = 1;

            m_Data = Memory::Buffer<T>(val);
        }

        Tensor(std::initializer_list<T> list) {
            if (!(typeid(T) == typeid(FP32) || typeid(T) == typeid(FP64) || typeid(T) == typeid(I32) || typeid(T) == typeid(I64))) {
                LOG_ERROR("[ERROR] (Tensor::Tensor) Unsupported tensor type!");
            }

            m_Shape = TensorShape{list.size()};
            m_Size = list.size();
            m_Data = Memory::Buffer<T>(m_Size);

            _calculate_index_offsets(m_Shape);

            std::memcpy(m_Data.begin().host_ptr(), list.begin(), m_Size * sizeof(T));
        }

        Tensor(std::initializer_list<std::initializer_list<T>> list) {
            if (!(typeid(T) == typeid(FP32) || typeid(T) == typeid(FP64) || typeid(T) == typeid(I32) || typeid(T) == typeid(I64))) {
                LOG_ERROR("[ERROR] (Tensor::Tensor) Unsupported tensor type!");
            }

            auto first = list.begin();
            m_Shape = TensorShape{list.size(), first->size()};
            m_Size = list.size() * first->size();
            m_Data = Memory::Buffer<T>(m_Size);

            _calculate_index_offsets(m_Shape);

            uint32_t i = 0;
            for (auto rank_1_tensor : list) {
                if (rank_1_tensor.size() != m_Shape[1]) {
                    LOG_ERROR("[ERROR] (Tensor::Tensor) Tensor sizes dont match!");
                }

                std::memcpy(m_Data.begin().host_ptr() + (i * first->size()), rank_1_tensor.begin(), m_Shape[1] * sizeof(T));

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
            m_Data = Memory::Buffer<T>(m_Size);

            _calculate_index_offsets(m_Shape);

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

                    std::memcpy((m_Data.begin().host_ptr() + (i * m_Strides[0] + j * m_Strides[1])), rank_1_tensor.begin(), m_Shape[2] * sizeof(T));
                    
                    j++;
                }

                i++;
            }
        }

        Tensor(const Tensor<T>& t) : 
        m_Data(t.m_Data), 
        m_Size(t.m_Size), 
        m_Shape(t.m_Shape), 
        m_Strides(t.m_Strides) {
        };

        Tensor(Tensor<T>&& other) : 
        m_Data(std::move(other.m_Data)),
        m_Size(other.m_Size),
        m_Shape(other.m_Shape),
        m_Strides(other.m_Strides) {
        }

        Tensor<T>& operator=(const Tensor<T>& other) {
            m_Data = other.m_Data;
            m_Size = other.m_Size;
            m_Shape = other.m_Shape;
            m_Strides = other.m_Strides;
        }

        Tensor<T>& operator=(Tensor<T>&& other) {
            m_Data = other.m_Data;
            m_Size = other.m_Size;
            m_Shape = other.m_Shape;
            m_Strides = other.m_Strides;
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

            if (rank() == 1) {
                return Tensor(m_Data[_index({index})]);
            }

            typename Memory::Buffer<T>::BufferIterator start = m_Data.begin() + (index * m_Shape[0]);
            typename Memory::Buffer<T>::BufferIterator end = m_Data.begin() + ((index + 1) * m_Shape[0]);

            Memory::Buffer<T> result_buffer(start, end);
            TensorShape result_shape(m_Shape.begin() + 1, m_Shape.end());

            return Tensor<T>(result_buffer, result_shape);
        }

        // T& operator() (uint32_t x, uint32_t y = 0, uint32_t z = 0) {
        //     if (rank() == 3 && x > m_Shape[rank() - 3]) {
        //         LOG_ERROR("[ERROR] (Tensor::operator()) Invalid access x index!");
        //     }

        //     if (rank() != 1 && y > m_Shape[rank() - 2]) {
        //         LOG_ERROR("[ERROR] (Tensor::operator()) Invalid access y index!");
        //     }

        //     if (z > m_Shape[rank() - 1]) {
        //         LOG_ERROR("[ERROR] (Tensor::operator()) Invalid access z index!");
        //     }

        //     return m_Data.get()[_index({x, y, z})];
        // }
        
        void set(T val, std::initializer_list<size_t> indices) {
            m_Data[_index(indices)] = val;
        }

        void set(T val, size_t internal_index) {
            if (internal_index > m_Size-1) {
                LOG_ERROR("[ERROR] (Tensor::set) Invalid internal access index!");
            }

            m_Data[internal_index] = val;
        }

        T get(std::initializer_list<size_t> indices) const {
            return m_Data[_index(indices)];
        }

        T get(uint32_t internal_index) const {
            if (internal_index > m_Size - 1) {
                LOG_ERROR("[ERROR] (Tensor::get) Invalid internal access index! {}", internal_index);
            }

            return m_Data[internal_index % m_Data.GetSize()];
        }

        // If possible, broadcast to a specific shape
        // Broadcast tensors do NOT own the underlying memory
        Tensor<T> broadcast(TensorShape broadcast_shape) const {
            size_t broadcast_rank       = broadcast_shape.size();
            size_t current_rank         = rank();
            TensorShape original_shape  = m_Shape;

            if (broadcast_rank < current_rank) {
                LOG_ERROR("[ERROR] (Tensor::broadcast) Invalid broadcast shape!");
            }

            original_shape.insert(original_shape.begin(), broadcast_rank - current_rank, 1);
            TensorShape new_shape = original_shape;

            for (size_t i = 0; i < broadcast_rank; ++i) {
                if (new_shape[i] != broadcast_shape[i] && new_shape[i] != 1){
                    LOG_ERROR("[ERROR] (Tensor::broadcast) Cannot broadcast a {}-rank shape to a {}-rank shape", current_rank, broadcast_rank);
                }

                new_shape[i] = std::max(new_shape[i], broadcast_shape[i]);
            }

            size_t new_size = std::accumulate(new_shape.begin(), new_shape.end(), 1, std::multiplies<T>());

            return Tensor<T>(m_Data, new_shape, original_shape);
        }

        inline size_t size() const { return m_Size; }
        inline size_t rank() const { return m_Shape.size(); }
        inline TensorShape shape() const { return m_Shape; }
        inline Memory::Buffer<T>& data() { return m_Data; }
        inline FP64 identifier() const { 
            size_t _rank = rank();

            for (auto element : m_Shape) {
                if (element == 1) _rank--;
            }

            FP64 id = ((FP64)m_Size / (m_Size + 3)) * _rank;

            return id; 
        }

        // TENSOR OPERATIONS --------------------------------------------------------------------------------------------------------------
        
        template <typename B>
        friend std::ostream& operator<<(std::ostream& out, const Tensor<B>& tensor);
};

template <typename T>
std::ostream& operator<<(std::ostream& out, const Tensor<T>& tensor) {

    if (tensor.rank() > 3) {
        LOG_ERROR("[ERROR] (Tensor::operator<<) Cannot display tensors whose rank is bigger than 3");
    }

    switch (tensor.rank()) {
        case 1: {
            out << "{";

            for (size_t i = 0; i < tensor.m_Shape[0] - 1; ++i) {
                out << tensor.get({i}) << ", ";
            }

            out << tensor.get({tensor.m_Shape[0] - 1});

            out << "}";
            break;
        }

        case 2: {
            out << "{" << std::endl;
            for (size_t j = 0; j < tensor.m_Shape[0]; ++j) {
                out << "    ";
                out << "{";

                for (size_t i = 0; i < tensor.m_Shape[1] - 1; ++i) {
                    out << tensor.get({j, i}) << ", ";
                }

                out << tensor.get({j, tensor.m_Shape[1] - 1}) << "}";
                out << std::endl;
            }
            out << "}";
            break;
        }

        case 3: {
            out << "{" << std::endl;

            for (size_t k = 0; k < tensor.m_Shape[0]; ++k) {

                out << "    ";
                out << "{" << std::endl;
                out << "    ";

                for (size_t j = 0; j < tensor.m_Shape[1]; ++j) {
                    out << "    ";
                    out << "{";

                    for (size_t i = 0; i < tensor.m_Shape[2] - 1; ++i) {
                        out << tensor.get({k, j, i}) << ", ";
                    }

                    out << tensor.get({k, j, tensor.m_Shape[2] - 1}) << "}";
                    out << std::endl;
                    out << "    ";
                }

                out << "}" << std::endl;
            }

            out << "}";
            break;
        }
    }

    return out;
}

#endif // TENSOR_H
