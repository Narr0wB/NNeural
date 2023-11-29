
#include <iostream>
#include "tensor/Tensor.h"
#include "tensor/TensorOperations.h"
#include "utils/Log.h"
#include "model/Layer.h"

template <typename T>
T ReLU(T in) {
    return in > 0 ? in : 0;
}

// TODO:
// Implement the gpu kernels for forward and backward propagation
// Check performance for host allocated device accessible memory (hipHostMalloc)

int main(void) {
    Log::Init();
    
    Tensor<FP32> t = {
        {2, 3, 4},
        {1, 1, 1},
        {1, 1, 1}
    };

    Tensor<FP32> t2 = {1, 2, 3};

    

    //Layer<FP32, Hardware::CPU> layer({1}, {2}, ReLU<FP32>);

    //LOG_INFO("lil meech {}", t.broadcast({10, 2, 3})(9, 0, 2));

    // assert(t.size() == 2 * 3);
    // assert(t2.size() == 3 * 4);gdb

    LOG_INFO("lil meech {}", t2(1));

    // LOG_INFO("{}", a(0, 1, 1));


    Tensor<FP32> added = scale(t, 0.5f);
    
    // LOG_INFO("Multiplying a ({}, {}, {}) Tensor with a ({}, {}, {}) Tensor resulting in a rank {} Tensor", 1, 1, t2.shape()[0], 1, t.shape()[0], t.shape()[1], multiplied.rank());

    std::cout << added << std::endl;
    std::cout << added.shape() << std::endl;



    // for (int i = 0; i < 2 * 4; i++) {
    //     LOG_INFO("{}", multiplied.data()[i]);
    // }

    return 0;
}