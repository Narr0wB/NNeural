
#include "Operations.h" 

void matmul(DataType type, void* _result, void* _m1_data, void* _m2_data, MatrixShape m1_shape, MatrixShape m2_shape) {
    if (m1_shape.cols != m2_shape.rows) {
        LOG_ERROR("[ERROR] (operations/matmul) Invalid input matrices!");
    }

    switch (type) {
        case DataType::FP32: {

            FP32* result  = (FP32*)_result;
            FP32* m1_data = (FP32*)_m1_data;
            FP32* m2_data = (FP32*)_m2_data;

            // TODO: IMPLEMENT MULTITHREADING (CPU, GPU)

            for (uint32_t i = 0; i < m1_shape.rows; ++i) {
                for (uint32_t j = 0; j < m2_shape.cols; ++j) {
                    FP32 temp_sum = 0;   

                    for (uint32_t k = 0; k < m1_shape.cols; ++k) {
                        temp_sum += m1_data[IX(i, k, m1_shape.cols)] * m2_data[IX(k, j, m2_shape.cols)];
                    }
                    
                    result[IX(i, j, m2_shape.cols)] = temp_sum;
                }
            }

            

            break;
        }

        case DataType::FP64: {
            
            FP64* result  = (FP64*)_result;
            FP64* m1_data = (FP64*)_m1_data;
            FP64* m2_data = (FP64*)_m2_data;

            // TODO: IMPLEMENT MULTITHREADING (CPU, GPU)

            for (uint32_t i = 0; i < m1_shape.rows; ++i) {
                for (uint32_t j = 0; j < m2_shape.cols; ++j) {
                    FP64 temp_sum = 0;

                    for (uint32_t k = 0; k < m1_shape.cols; ++k) {
                        temp_sum += m1_data[IX(i, k, m1_shape.cols)] * m2_data[IX(k, j, m2_shape.cols)];
                    }

                    result[IX(i, j, m2_shape.cols)] = temp_sum;
                }
            }
            
            break;
        }

        case DataType::I32: {

            I32* result  = (I32*)_result;
            I32* m1_data = (I32*)_m1_data;
            I32* m2_data = (I32*)_m2_data;

            // TODO: IMPLEMENT MULTITHREADING (CPU, GPU)

            for (uint32_t i = 0; i < m1_shape.rows; ++i) {
                for (uint32_t j = 0; j < m2_shape.cols; ++j) {
                    I32 temp_sum = 0;

                    for (uint32_t k = 0; k < m1_shape.cols; ++k) {
                        temp_sum += m1_data[IX(i, k, m1_shape.cols)] * m2_data[IX(k, j, m2_shape.cols)];
                    }

                    result[IX(i, j, m2_shape.cols)] = temp_sum;
                }
            }

            break;
        }

        case DataType::I64: {

            I64* result  = (I64*)_result;
            I64* m1_data = (I64*)_m1_data;
            I64* m2_data = (I64*)_m2_data;

            // TODO: IMPLEMENT MULTITHREADING (CPU, GPU)

            for (uint32_t i = 0; i < m1_shape.rows; ++i) {
                for (uint32_t j = 0; j < m2_shape.cols; ++j) {
                    I64 temp_sum = 0;

                    for (uint32_t k = 0; k < m1_shape.cols; ++k) {
                        temp_sum += m1_data[IX(i, k, m1_shape.cols)] * m2_data[IX(k, j, m2_shape.cols)];
                    }

                    result[IX(i, j, m2_shape.cols)] = temp_sum;
                }
            }

            break;
        }

        default: {
            LOG_ERROR("[ERROR] (operations/matmul) Invalid DataType enum!");
        }
    }
}