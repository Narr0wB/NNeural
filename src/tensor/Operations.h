
#ifndef OPERATIONS_H
#define OPERATIONS_H

#include "../Types.h"
#include "../utils/Log.h"

#define IX(x, y, N) (x * N) + y

typedef struct {
    uint32_t rows;
    uint32_t cols;
} MatrixShape;

void matmul(DataType type, void* result, void* m1_data, void* m2_data, MatrixShape m1_shape, MatrixShape m2_shape);

#endif // OPERATIONS_H