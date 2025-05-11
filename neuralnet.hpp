#include "layer.hpp"
#include <vector>

using namespace Eigen;

class NeuralNet {
  private:
    std::vector<Layer> layers;

  public:
    NeuralNet() {}

    void runNeuralNet(const VectorXd &input) {
        layers[0].forward(input);
        for (size_t i = 0; i < layers.size(); i++) {
            layers[i].forward(layers[i - 1].getActivations());
        }
    }

    VectorXd getPrediction() {
        Layer predictionLayer = layers.back();
        return predictionLayer.getActivations();
    }
};
