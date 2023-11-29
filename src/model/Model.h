
#ifndef MODEL_H
#define MODEL_H

#include "Layer.h"

template <typename T, Hardware H = Hardware::CPU>
class Model {
    private:
        std::vector<Layer<T, H>> m_Layers;

    public:
        Model(std::initializer_list<Layer<T, H>> list) : m_Layers(list) {}

        void train();

        void save(std::string path);
        void load(std::string path);

        Tensor<T> forward(Tensor<T> input) {
            switch (H) {
                case Hardware::CPU: {
                    return _high_throughput_tensor_CPU(input);
                }
                case Hardware::GPU: {
                    return _high_throughput_tensor_GPU(input);
                }

                default:
                    return Tensor<T>();
            }
        }

        Tensor<T> operator() (Tensor<T> input) {
            return forward(input);
        }
};

#endif // MODEL_H