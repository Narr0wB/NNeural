
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

    //LOG_INFO("lil meech {}", t.broadcast({10, 2, 3})(9, 0, 2));

    // assert(t.size() == 2 * 3);
    // assert(t2.size() == 3 * 4);gdb

    // auto a = t2.broadcast({3, 3, 3});
    // std::cout << a << std::endl;

    // LOG_INFO("{}", a(0, 1, 1));

    Tensor<FP32> multiplied = tensormul(t, t2);
    
    // LOG_INFO("Multiplying a ({}, {}, {}) Tensor with a ({}, {}, {}) Tensor resulting in a rank {} Tensor", 1, 1, t2.shape()[0], 1, t.shape()[0], t.shape()[1], multiplied.rank());

    std::cout << t << std::endl;
    std::cout << t2 << std::endl;
    std::cout << multiplied << std::endl;

    // for (int i = 0; i < 2 * 4; i++) {
    //     LOG_INFO("{}", multiplied.data()[i]);
    // }

    return 0;
}