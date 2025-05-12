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

    void addLayer(int in, int neurons) {
        layers.emplace_back(in, neurons);
    }

    void forward(const VectorXd& input) {
        layers[0].forward(input);
        for (size_t i = 1; i < layers.size(); i++) {
            layers[i].forward(layers[i - 1].getActivations());
        }
    }

    void backward(const VectorXd& input, const VectorXd& target, double lr) {
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
            VectorXd in;
            if (i == 0) {
                in = input;
            } 
            else {
                in = layers[i - 1].getActivations();
            }

            layers[i].updateWeights(in, lr);
        }
    }

    void train(const std::vector<VectorXd>& data, const std::vector<VectorXd>& labels, double lr, int epochs) {
        std::cout << "Training..." << std::endl;
        for (int epoch = 0; epoch < epochs; epoch++) {
            double totalLoss = 0.0;
    
            for (size_t i = 0; i < data.size(); i++) {
                VectorXd input = data[i];

                forward(input);
    
                VectorXd output = layers.back().getActivations();
                totalLoss += MSE(labels[i], output);
    
                backward(data[i], labels[i], lr);

                if (i + 1 % 100 == 0) {
                    std::cout << "Training sample " << i + 1 << "/" << data.size() << "..." << std::endl;
                }
            }
            totalLoss /= data.size();
            std::cout << "Epoch " << epoch + 1 << "/" << epochs << ", Loss: " << totalLoss << std::endl;
        }
        std::cout << "Training completed." << std::endl;
    }
    
    double test(const std::vector<VectorXd>& data, const std::vector<VectorXd>& labels) {
        std::cout << "Starting testing..." << std::endl;

        int correct = 0;
    
        for (size_t i = 0; i < data.size(); i++) {
            forward(data[i]);
    
            VectorXd output = layers.back().getActivations();
    
            int predicted = std::distance(output.data(), std::max_element(output.data(), output.data() + output.size()));
            int actual = std::distance(labels[i].data(), std::max_element(labels[i].data(), labels[i].data() + labels[i].size()));
    
            if (predicted == actual) {
                correct++;
            }
            if (i + 1 % 100 == 0) {
                std::cout << "Testing sample " << i + 1 << "/" << data.size() << "..." << std::endl;
            }
        }

        int accuracy =  (double)correct / data.size() * 100.0;
        std::cout << "Testing completed. Accuracy: " << accuracy << "%" << std::endl;
        return accuracy;
    }
};