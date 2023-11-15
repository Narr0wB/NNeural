
#ifndef TENSOR_OPERATIONS_H
#define TENSOR_OPERATIONS_H

#include "Tensor.h"

// Computes the matrix multiplications between two tensors (any tensor with rank bigger than 2 is considered a collection of matrices)
// Broadcasting: Allows for broadcasting
template <typename T>
Tensor<T> tensormul(Tensor<T>& a, Tensor<T>& b) {
    uint32_t a_3_rank = a.rank() == 3 ? a.shape()[0] : 1;
    uint32_t b_3_rank = b.rank() == 3 ? b.shape()[0] : 1;

    uint32_t a_2_rank = a.rank() == 1 ? 1 : a.shape()[a.rank() - 2];
    uint32_t b_2_rank = b.rank() == 1 ? 1 : b.shape()[b.rank() - 2];

    uint32_t a_1_rank = a.shape()[a.rank() - 1];
    uint32_t b_1_rank = b.shape()[b.rank() - 1];

    Tensor<T> result(a_3_rank, a_2_rank, b_1_rank);

    // Do not broadcast if:
    if (
        a_3_rank != b_3_rank && // The tensors' highest rank is different AND
        b_3_rank != 1        && 
        a_3_rank != 1        && // Both tensors' highest rank is different from 1 OR
        a_1_rank == b_2_rank    // A's lowest rank is different from B's second rank
        ) { 
        for (uint32_t r = 0; r < a_3_rank; ++r) {
            for (uint32_t i = 0; i < a_2_rank; ++i) {
                for (uint32_t j = 0; j < b_1_rank; ++j) {
                    T temp_sum = 0;

                    for (uint32_t k = 0; k < a_1_rank; ++k) {
                        temp_sum += (a(r, i, k) * b(r, k, j));
                    }

                    result.set(temp_sum, r, i, j);
                }
            }
        }
    }

    // If the previous conditions are not met, then try to broadcast the B tensor into a compatible shape
    // In case the tensor could not be broadcasted, then the broadcast function will log an error and stop execution
    else {
        Tensor<T>& original = a;
        BroadcastTensor<T> broadcasted = b.broadcast({a_3_rank, a_1_rank, b_1_rank});

        for (uint32_t r = 0; r < a_3_rank; ++r) {
            for (uint32_t i = 0; i < a_2_rank; ++i) {
                for (uint32_t j = 0; j < b_1_rank; ++j) {
                    T temp_sum = 0;

                    for (uint32_t k = 0; k < a_1_rank; ++k) {
                        temp_sum += (original(r, i, k) * broadcasted(r, k, j));
                    }

                    result.set(temp_sum, r, i, j);
                }
            }
        }    
    }


    return result;
}

// Computes the sum of two tensors in the form of: a + b
// Broadcasting: Allows for broadcasting
template <typename T>
Tensor<T> add(Tensor<T> a, Tensor<T> b) {
    
    if (a.identifier() == b.identifier()) {
        Tensor<T> result(a.shape());

        for (size_t i = 0; i < result.size(); ++i) {
            result.set(a.get(i) + b.get(i), i);
        }

        return result;
    }

    if (a.identifier() > b.identifier()) {
        Tensor<T> result(a.shape());
        BroadcastTensor<T> b_broadcasted = b.broadcast(a.shape());

        for (size_t i = 0; i < result.size(); ++i) {
            result.set(a.get(i) + b_broadcasted.get(i), i);
        }

        return result;
    }

    else {
        Tensor<T> result(b.shape());
        BroadcastTensor<T> a_broadcasted = a.broadcast(b.shape());

        for (size_t i = 0; i < result.size(); ++i) {
            result.set(b.get(i) + a_broadcasted.get(i), i);
        }

        return result;
    }
}

// Computes the difference between two tensors in the form of: a - b
// Broadcasting: Allows for broadcasting
template <typename T>
Tensor<T> subtract(Tensor<T> a, Tensor<T> b) {
    
    if (a.identifier() == b.identifier()) {
        Tensor<T> result(a.shape());

        for (size_t i = 0; i < result.size(); ++i) {
            result.set(a.get(i) - b.get(i), i);
        }

        return result;
    }

    if (a.identifier() > b.identifier()) {
        Tensor<T> result(a.shape());
        BroadcastTensor<T> b_broadcasted = b.broadcast(a.shape());

        for (size_t i = 0; i < result.size(); ++i) {
            result.set(a.get(i) - b_broadcasted.get(i), i);
        }

        return result;
    }

    else {
        Tensor<T> result(b.shape());
        BroadcastTensor<T> a_broadcasted = a.broadcast(b.shape());

        for (size_t i = 0; i < result.size(); ++i) {
            result.set(a_broadcasted.get(i) - b.get(i), i);
        }

        return result;
    }
}

// Computes the element-wise scaling of a given tensor
// Broadcasting: Broadcasting not needed
template <typename T>
Tensor<T> scale(Tensor<T> a, T scalar) {
    Tensor<T> result(a.shape());

    for (size_t i = 0; i < a.size(); ++i) {
        result.set(a.get(i) * scalar, i);
    }

    return result;
}


// Computes the hadamard product between two tensors (element-wise product)
// Broadcasting: Allows for broadcasting
template <typename T>
Tensor<T> hadamard(Tensor<T> a, Tensor<T> b) {
    if (a.identifier() == b.identifier()) {
        Tensor<T> result(a.shape());

        for (size_t i = 0; i < a.size(); ++i) {
            result.set(a.get(i) * b.get(i), i);
        }

        return result;
    }

    if (a.identifier() > b.identifier()) {
        Tensor<T> result(a.shape());
        BroadcastTensor<T> b_broadcasted = b.broadcast(a.shape());

        for (size_t i = 0; i < result.size(); ++i) {
            result.set(a.get(i) * b_broadcasted.get(i), i);
        }

        return result;
    }
    else {
        Tensor<T> result(b.shape());
        BroadcastTensor<T> a_broadcasted = a.broadcast(b.shape());

        for (size_t i = 0; i < result.size(); ++i) {
            result.set(b.get(i) * a_broadcasted.get(i), i);
        }

        return result;
    }
}

#endif // TENSOR_OPERATIONS_H