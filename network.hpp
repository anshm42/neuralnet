#include <iostream>
#include <vector>
#include "layer.hpp"

class Network {
private:
    std::vector<Layer> layers;

public:
    Network(int in, int out) {
        layers.emplace_back(in, out);
    }

    void addLayer(int neurons) {
        int in = layers.back().getWeights().rows();
        layers.emplace_back(in, neurons);
    }

    void forward(VectorXd& input) {
        layers[0].forward(input);
        for (int i = 1; i < layers.size(); i++) {
            layers[i].forward(layers[i - 1].getActivations());
        }
    }
};