
#include <iostream>
#include "tensor/Tensor.h"
#include "utils/Log.h"


int main(void) {
    Log::Init();
    
    Tensor<FLOAT32> t{{2, 3, 4}};

    std::cout << t.getSize();

    return 0;
}