
#include <iostream>
#include "tensor/Tensor.h"
#include "utils/Log.h"


int main(void) {
    Log::Init();
    
    Tensor<FLOAT32> t = {
       {{2, 3, 4},
        {1, 1, 1}},

        {{5, 3, 8},
        {6, 10, 7}},
    };

    assert(t.size() == 2 * 2 * 3);
    assert(t[0].size() == 2 * 3);



    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            for (int k = 0; k < 3; ++k) {
                LOG_INFO("TENSOR {}", (FLOAT32)t[i][j][k]);
            }
        }
    }

    return 0;
}