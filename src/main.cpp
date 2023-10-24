
#include <iostream>
#include "tensor/Tensor.h"
#include "utils/Log.h"


int main(void) {
    Log::Init();
    
    Tensor<FP32> t = {
        {2, 3, 4},
        {1, 1, 1},
    };

    Tensor<FP32> t2 = {1, 2, 3};

    // assert(t.size() == 2 * 3);
    // assert(t2.size() == 3 * 4);


    Tensor<FP32> multiplied = tensormul(t, t2);
    
    LOG_INFO("Multiplying a ({}, {}, {}) Tensor with a ({}, {}, {}) Tensor resulting in a rank {} Tensor", 1, 1, t2.shape()[0], 1, t.shape()[0], t.shape()[1], multiplied.rank());

    for (int i = 0; i < 1; ++i) {
        for (int j = 0; j < 1; ++j) {
            for (int k = 0; k < multiplied.shape()[0]; ++k) {
                LOG_INFO("TENSOR ({}, {}, {}) = {}", i, j, k, (FP32)multiplied[k]);
            }
        }
    }

    // for (int i = 0; i < 2 * 4; i++) {
    //     LOG_INFO("{}", multiplied.data()[i]);
    // }

    return 0;
}