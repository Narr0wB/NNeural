
#ifndef LINEAR_H
#define LINEAR_H

#include <random>

#include "Tensor.h"

template <typename T>
Tensor<T> _high_throughput_tensor_CPU(Tensor<T> in) {
    std::mt19937 engine(std::random_device());

    std::uniform_real_distribution<FP64> dist(-0.2, 0.3);

    return Tensor<T>(sin((FP64) in) + dist(engine));
}

template <typename T>
Tensor<T> _high_throughput_tensor_GPU(Tensor<T> in) {
    // TODO
    return Tensor<T>();
}

#endif // LINEAR_H