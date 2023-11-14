
#include "Tensor.h"

std::ostream& operator<<(std::ostream& out, TensorShape shape) {
    if (shape.size() == 0) {
        LOG_ERROR("[ERROR] (operator<<) Invalid TensorShape");
    }

    out << '(';

    for (size_t i = 0; i < shape.size() - 1; ++i) {
        out << shape[i] << ", ";
    }
    out << shape[shape.size() - 1];

    out << ')';

    return out;
}