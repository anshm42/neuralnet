#include "functions.hpp"
#include <Eigen/Dense>

using namespace Eigen;

class Layer {
  private:
    MatrixXd weights;
    VectorXd biases;

    VectorXd delta;

    VectorXd last_input, last_z, last_a;

  public:
    Layer(int inSize, int outSize) {
        weights = MatrixXd::Random(inSize, outSize);
        biases = VectorXd::Zero(outSize);
    }

    VectorXd forward(const VectorXd &input) {
        last_input = input;
        last_z = (weights * input) + biases;
        last_a = sigmoid(last_z);
        return last_a;
    }

    VectorXd backward(const VectorXd &gradOutput, float lr) {
        VectorXd delta = gradOutput.array() * dsigmoid(last_a).array();
        MatrixXd dW = delta * last_input.transpose();
        VectorXd db = delta;
        weights.noalias() -= lr * dW;
        biases -= lr * db;
        return weights.transpose() * delta;
    }
};
