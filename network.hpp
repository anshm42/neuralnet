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

    void backward(VectorXd& input, VectorXd& target, double lr) {
        VectorXd output = layers.back().getActivations();
        VectorXd error = dMSE(output, target);

        VectorXd delta = error.cwiseProduct(dSigmoid(output));

        layers.back().setDelta(delta);

        for (int i = layers.size() - 2; i >= 0; i--) {
            MatrixXd nextWeights = layers[i + 1].getWeights();
            VectorXd nextDelta = layers[i + 1].getDelta();

            VectorXd hiddenError = nextWeights * nextDelta;
            VectorXd dHidden = dSigmoid(layers[i].getActivations());
            VectorXd hiddenDelta = hiddenError.cwiseProduct(dHidden);
            layers[i].setDelta(hiddenDelta);
        }

        for (int i = 0; i < layers.size(); i++) {
            VectorXd input_;
            if (i == 0) {
                input_ = input;
            } 
            else {
                input_ = layers[i - 1].getActivations();
            }

            layers[i].updateWeights(input_, lr);
        }
    }
};