
#ifndef LAYER_H
#define LAYER_H

#include "../tensor/Tensor.h"
#include "../tensor/TensorOperations.h"
#include "../tensor/Hardware.h"

#define LA_SIGMOID 1
#define LA_RELU    2
#define LA_MISH    3

template <typename T, Hardware H = Hardware::CPU>
class Layer {
    private:
        TensorShape m_InputShape;   // Shape of the layer's input
        TensorShape m_OutputShape;  // Shape of the layer's output

        Tensor<T> m_W;  // Weights of the layer
        Tensor<T> m_B;  // Biases of the layer

        Tensor<T> m_Z;  // Placeholder for the tensor product between the previous layer's input and the current layer's weight tensor
        Tensor<T> m_A;  // Current layer's output tensor

        Tensor<T> m_dA; // Current layer's delta with respect to the cost function

        T (*m_Activation)(T val);   // The activation function of the layer (CPU ONLY)
        int m_GActivation;          // The activation function index (in a list of activation functions defined in the gpu) (GPU ONLY)

    public:

        // Enable this constructor only if the layer is a CPU layer
        template <Hardware B = H, EnableIf<B == Hardware::CPU> = 0> 
        Layer(TensorShape in_shape, TensorShape out_shape, T (*activation)(T val) = NULL) : m_InputShape(in_shape), m_OutputShape(out_shape), m_Activation(activation) {
            if (in_shape.size() != 1 || out_shape.size() != 1) {
                LOG_ERROR("(Layer::Layer) Invalid input/output shape! (input and output shapes have to be 1-dimentional)");
            }

            m_W  = std::move(Tensor<T>(out_shape[0], in_shape[0]));
            m_B  = std::move(Tensor<T>(out_shape));
            m_Z  = std::move(Tensor<T>(out_shape));
            m_A  = std::move(Tensor<T>(out_shape));
        }

        // Enable this constructor only if the layer is a GPU layer
        template <Hardware B = H, EnableIf<B == Hardware::GPU> = 0>
        Layer(TensorShape in_shape, TensorShape out_shape, int function_operator) : m_InputShape(in_shape), m_OutputShape(out_shape), m_GActivation(function_operator) {
            if (in_shape.size() != 1 || out_shape.size() != 1) {
                LOG_ERROR("(Layer::Layer) Invalid input/output shape! (input and output shapes have to be 1-dimentional)");
            }

            m_W  = std::move(Tensor<T>(out_shape[0], in_shape[0]));
            m_B  = std::move(Tensor<T>(out_shape));
            m_Z  = std::move(Tensor<T>(out_shape));
            m_A  = std::move(Tensor<T>(out_shape));
        }


        // Forward propagate given the previous' layer output (n is the previous layer, m is the current, p is the next layer)
        Tensor<T>& forward(const Tensor<T>& An) {
            if (An.shape() != m_InputShape) {
                LOG_ERROR("(Layer::forward) Incompatible input shape!");
            }

            switch (H) {
                case Hardware::CPU: {
                    _tensor_forward(m_W, An, m_B, m_Z);
                    _tensor_apply(m_Activation, m_Z, m_A)
                    break;
                }

                case Hardware::GPU: {
                    _gpu_tensor_forward(m_W, An, m_B, m_Z);
                    _gpu_tensor_apply(m_GActivation, m_Z, m_A);
                    break;
                }

                default:
                    break;
            }
            
            return m_A;
        }

        // Back propagate the error given the next layer's delta
        Tensor<T>& backward(const Tensor<T>& dA) {

        }
};

template <typename T>
void _tensor_forward(Tensor<T>& Wn, Tensor<T>& An, Tensor<T>& Bn, Tensor<T>& result) {

}

#endif // LAYER_H