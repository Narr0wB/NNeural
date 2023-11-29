
#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <iostream>
#include <memory>
#include <functional>
#include <numeric>

#include "../types.h"
#include "../utils/Log.h"
#include "../utils/Hardware.h"
#include "../utils/Memory.h"

#define MAX_RANK 3

typedef std::vector<uint32_t> TensorShape;
std::ostream& operator<<(std::ostream& out, TensorShape shape);

template <typename T>
class BroadcastTensor {
    private:
        TensorShape m_Shape;
        TensorShape m_OriginalShape;

        size_t m_Size;
        size_t m_OriginalSize;
        std::shared_ptr<T[]> m_Data; 

        // Get data's index in the internal buffer
        size_t _index(std::initializer_list<uint32_t> parameters) {
            size_t index = 0;
            if (parameters.size() != m_OriginalShape.size()) {
                //LOG_ERROR("DEBUG: EROR");
            }

            for (int i = 0; i < m_OriginalShape.size(); ++i) {
                if (m_OriginalShape[rank() - i - 1] != 1) {
                    auto current_parameter = *(parameters.end() - i - 1);
                
                    index += current_parameter * std::accumulate(m_OriginalShape.end() - i, m_OriginalShape.end(), 1, std::multiplies<T>());
                }
            }

            return index;
        }

    public:

        BroadcastTensor(std::shared_ptr<T[]>& original_data, size_t size, size_t original_size, TensorShape& shape, TensorShape& original_shape) : m_Data(original_data), m_Shape(shape), m_OriginalShape(original_shape), m_Size(size), m_OriginalSize(original_size) {}

        T operator() (uint32_t x, uint32_t y, uint32_t z) {
            return m_Data[_index({x, y, z})];
        }

        T get(uint32_t internal_index) {
            if (internal_index > m_Size - 1) {
                LOG_ERROR("[ERROR] (Tensor::get) Invalid internal access index!");
            }

            return m_Data[internal_index % m_OriginalSize];
        }

        inline size_t rank() { return m_Shape.size(); }
        inline FP64 identifier() { return ((FP64)m_Size / (m_Size + 3)) * rank(); }

        template <typename B>
        friend std::ostream& operator<<(std::ostream& out, BroadcastTensor<B>& tensor) {
            switch (tensor.rank()) {
                case 1: {
                    out << "{";

                    for (uint32_t i = 0; i < tensor.m_Shape[0] - 1; ++i) {
                        out << tensor.m_Data[tensor._index({i})] << ", ";
                    }

                    out << tensor.m_Data[tensor._index({tensor.m_Shape[0] - 1})];

                    out << "}";
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
                    out << "}";
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

                    out << "}";
                    break;
                }
            }

            return out;
        }
};

template <typename T, Hardware H = Hardware::CPU>
class Tensor {
    private:
        TensorShape m_Shape;
        size_t m_Size = 1;
        std::shared_ptr<T> m_Data;

        // Get data's index in the internal buffer
        size_t _index(std::initializer_list<uint32_t> parameters) {
            size_t index = 0;
            if (parameters.size() != m_Shape.size()) {
                //LOG_ERROR("DEBUG: EROR");
            }

            for (int i = 0; i < m_Shape.size(); ++i) {
                auto current_parameter = *(parameters.end() - i - 1);

                index += current_parameter * std::accumulate(m_Shape.end() - i, m_Shape.end(), 1, std::multiplies<T>());
            }

            return index;
        }

    public:
        Tensor() {
            m_Shape = TensorShape{0};
            m_Size = 0;
        }

        Tensor(TensorShape shape) : m_Shape(shape) {
            if (!(typeid(T) == typeid(FP32) || typeid(T) == typeid(FP64) || typeid(T) == typeid(I32) || typeid(T) == typeid(I64))) {
                LOG_ERROR("[ERROR] (Tensor::Tensor) Unsupported tensor type!");
            }
            
            if (shape.size() > 3) {
                LOG_ERROR("[ERROR] (Tensor::Tensor) The maximum supported rank is 3!");
            }

            for (uint32_t dimentions : shape) {
                m_Size *= dimentions;
            }

            m_Data = memory::_make_shared<T, H>(m_Size);
        }

        Tensor(T val) {
            if (!(typeid(T) == typeid(FP32) || typeid(T) == typeid(FP64) || typeid(T) == typeid(I32) || typeid(T) == typeid(I64))) {
                LOG_ERROR("[ERROR] (Tensor::Tensor) Unsupported tensor type!");
            }

            m_Shape = {1};
            m_Size = 1;

            m_Data = memory::_make_shared<T, H>(m_Size);

            if (H == Hardware::CPU) {
                m_Data.get()[0] = val;
            }
            else {
                _kernel_set(m_Data, 0, val); 
            }

        }

        Tensor(uint32_t x, uint32_t y = 0, uint32_t z = 0) {
            if (!(typeid(T) == typeid(FP32) || typeid(T) == typeid(FP64) || typeid(T) == typeid(I32) || typeid(T) == typeid(I64))) {
                LOG_ERROR("[ERROR] (Tensor::Tensor) Unsupported tensor type!");
            }

            // If one of the indices is one, then ignore said index as it does not provide useful information
            if (x == 1) {x = y; y = z; z = 0;}
            if (y == 1) {y = 0;}
            if (z == 1) {z = 0;}

            if (x == 0) {
                LOG_ERROR("[ERROR] (Tensor::Tensor) Invalid initialization index!");
            }

            if (y == 0 && z == 0) {
                m_Shape = TensorShape{x};
                m_Size = x;

                m_Data = memory::_make_shared<T, H>(m_Size);

                return;
            }

            if (z == 0) {
                m_Shape = TensorShape{x, y};
                m_Size = x*y;

                m_Data = memory::_make_shared<T, H>(m_Size);

                return;
            }

            m_Shape = TensorShape{x, y, z};
            m_Size = x*y*z;

            m_Data = memory::_make_shared<T, H>(m_Size);
        }

        Tensor(std::initializer_list<T> list) {
            if (!(typeid(T) == typeid(FP32) || typeid(T) == typeid(FP64) || typeid(T) == typeid(I32) || typeid(T) == typeid(I64))) {
                LOG_ERROR("[ERROR] (Tensor::Tensor) Unsupported tensor type!");
            }

            m_Shape = TensorShape{list.size()};
            m_Size = list.size();

            m_Data = memory::_make_shared<T, H>(m_Size * sizeof(T));

            if (H == Hardware::CPU) {
                std::memcpy(m_Data.get(), list.begin(), m_Size * sizeof(T));
            }
            else {
                hipMemcpyHtoD((hipDeviceptr_t) m_Data.get(), (void*) list.begin(), m_Size * sizeof(T));
            }
        }

        Tensor(std::initializer_list<std::initializer_list<T>> list) {
            if (!(typeid(T) == typeid(FP32) || typeid(T) == typeid(FP64) || typeid(T) == typeid(I32) || typeid(T) == typeid(I64))) {
                LOG_ERROR("[ERROR] (Tensor::Tensor) Unsupported tensor type!");
            }

            auto first = list.begin();
            m_Shape = TensorShape{list.size(), first->size()};
            m_Size = list.size() * first->size();


            m_Data = memory::_make_shared<T, H>(m_Size * sizeof(T));

            uint32_t i = 0;
            for (auto rank_1_tensor : list) {
                if (rank_1_tensor.size() != m_Shape[1]) {
                    LOG_ERROR("[ERROR] (Tensor::Tensor) Tensor sizes dont match!");
                }

                if (H == Hardware::CPU) {
                    std::memcpy(m_Data.get() + i * first->size(), rank_1_tensor.begin(), first->size() * sizeof(T));
                }
                else {
                    hipMemcpyHtoD((hipDeviceptr_t) m_Data.get() + i * first->size(), (void*) rank_1_tensor.begin(), first->size() * sizeof(T));
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

            m_Data = memory::_make_shared<T, H>(m_Size * sizeof(T));
            
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

                    if (H == Hardware::CPU) {
                        std::memcpy(m_Data.get() + i * first->size() + j * second->size(), rank_1_tensor.begin(), second->size() * sizeof(T));
                    }
                    else {
                        hipMemcpyHtoD(m_Data.get() + i * first->size() + j * second->size(), rank_1_tensor.begin(), second->size() * sizeof(T));
                    }
                    
                    j++;
                }

                i++;
            }
        }

        template <Hardware B = H, EnableIf<B == Hardware::CPU> = 0>
        operator T() { 
            if (m_Shape.size() != 1 && m_Shape[0] != 1) { 
                LOG_ERROR("[ERROR] Only scalar tensors can be converted to numbers!"); 
            } 

            return m_Data.get()[0];
        }

        template <Hardware B = H, EnableIf<B == Hardware::CPU> = 0>
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

        template <Hardware B = H, EnableIf<B == Hardware::CPU> = 0>
        T& operator() (uint32_t x, uint32_t y = 0, uint32_t z = 0) {
            if (rank() == 3 && x > m_Shape[rank() - 3]) {
                LOG_ERROR("[ERROR] (Tensor::operator()) Invalid access x index!");
            }

            if (rank() != 1 && y > m_Shape[rank() - 2]) {
                LOG_ERROR("[ERROR] (Tensor::operator()) Invalid access y index!");
            }

            if (z > m_Shape[rank() - 1]) {
                LOG_ERROR("[ERROR] (Tensor::operator()) Invalid access z index!");
            }

            return m_Data.get()[_index({x, y, z})];
        }
        
        template <Hardware B = H, EnableIf<B == Hardware::CPU> = 0>
        void set(T val, uint32_t x, uint32_t y, uint32_t z) {
            m_Data.get()[_index({x, y, z})] = val;
        }

        template <Hardware B = H, EnableIf<B == Hardware::CPU> = 0>
        void set(T val, uint32_t internal_index) {
            if (internal_index > m_Size-1) {
                LOG_ERROR("[ERROR] (Tensor::set) Invalid internal access index!");
            }

            m_Data.get()[internal_index] = val;
        }

        template <Hardware B = H, EnableIf<B == Hardware::CPU> = 0>
        T get(uint32_t internal_index) {
            if (internal_index > m_Size - 1) {
                LOG_ERROR("[ERROR] (Tensor::get) Invalid internal access index!");
            }

            return m_Data.get()[internal_index];
        }

        // If possible, broadcast to a specific shape
        template <Hardware B = H, EnableIf<B == Hardware::CPU> = 0>
        BroadcastTensor<T> broadcast(TensorShape broadcast_shape) {
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
                    LOG_ERROR("[ERROR] (Tensor::broadcast) Cannot broadcast a ({}, {}, {}) shape to a ({}, {}, {}) shape", broadcast_rank == 3 ? original_shape[0] : 1, broadcast_rank != 1 ? original_shape[broadcast_rank - 2] : 1, original_shape[broadcast_rank - 1], broadcast_rank == 3 ? broadcast_shape[0] : 1, broadcast_rank != 1 ? broadcast_shape[broadcast_rank - 2] : 1, broadcast_shape[broadcast_rank - 1]);
                }

                new_shape[i] = std::max(new_shape[i], broadcast_shape[i]);
            }

            size_t new_size = std::accumulate(new_shape.begin(), new_shape.end(), 1, std::multiplies<T>());

            return BroadcastTensor<T>(m_Data, new_size, m_Size, new_shape, original_shape);
        }

        inline size_t size() { return m_Size; }
        inline size_t rank() { return m_Shape.size(); }
        inline TensorShape shape() { return m_Shape; }
        inline T* data() { return m_Data.get(); }
        inline FP64 identifier() { 
            size_t _rank = rank();

            for (auto element : m_Shape) {
                if (element == 1) _rank--;
            }

            return ((FP64)m_Size / (m_Size + 3)) * _rank; 
        }

        // TENSOR OPERATIONS --------------------------------------------------------------------------------------------------------------
        
        template <typename B, Hardware G>
        friend std::ostream& operator<<(std::ostream& out, Tensor<B, G>& tensor);
};

template <typename T, Hardware G>
std::ostream& operator<<(std::ostream& out, Tensor<T, G>& tensor) {
            std::shared_ptr<T> m_Data;

            if (G == Hardware::CPU) {
                m_Data = tensor.m_Data;
            }
            else {
                T* host_ptr = new T[tensor.m_Size];

                hipMemcpyDtoH((void*) host_ptr, (hipDeviceptr_t) tensor.m_Data.get(), tensor.m_Size * sizeof(T));

                m_Data = std::shared_ptr<T>(host_ptr);
            }

            switch (tensor.rank()) {
                case 1: {
                    out << "{";

                    for (uint32_t i = 0; i < tensor.m_Shape[0] - 1; ++i) {
                        out << m_Data.get()[tensor._index({i})] << ", ";
                    }

                    out << m_Data.get()[tensor._index({tensor.m_Shape[0] - 1})];

                    out << "}";
                    break;
                }

                case 2: {
                    out << "{" << std::endl;
                    for (uint32_t j = 0; j < tensor.m_Shape[0]; ++j) {
                        out << "    ";
                        out << "{";

                        for (uint32_t i = 0; i < tensor.m_Shape[1] - 1; ++i) {
                            out << m_Data.get()[tensor._index({j, i})] << ", ";
                        }

                        out << m_Data.get()[tensor._index({j, tensor.m_Shape[1] - 1})] << "}";
                        out << std::endl;
                    }
                    out << "}";
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
                                out << m_Data.get()[tensor._index({k, j, i})] << ", ";
                            }

                            out << m_Data.get()[tensor._index({k, j, tensor.m_Shape[2] - 1})] << "}";
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