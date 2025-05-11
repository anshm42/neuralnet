#include <Eigen/Dense>

using namespace Eigen;

class Layer {
  private:
    VectorXd activations;

    MatrixXd weights;
    VectorXd biases;

  public:
    Layer(int inputSize, int outputSize) {
        weights = MatrixXd::Random(inputSize, outputSize);
        biases = VectorXd::Random(outputSize);
    }

    VectorXd sigmoid(const VectorXd &v) {
        return 1.0 / (1.0 + (-v.array()).exp());
    }

    VectorXd sigmoidDerivative(const VectorXd &v) {
        return sigmoid(v).array() * (1.0 - sigmoid(v).array());
    }

    void forward(const VectorXd &input) {
        activations = sigmoid((input * weights) + biases);
    }

    VectorXd getActivations() { return activations; }
};
