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

    VectorXd backward(const VectorXd &grad_output, float lr) {
        VectorXd delta = grad_output.array() * dsigmoid(last_a).array();
        MatrixXd dW = delta * last_input.transpose();
        VectorXd db = delta;
        weights.noalias() -= lr * dW;
        biases -= lr * db;
        return weights.transpose() * delta;
    }

    void set_delta(const VectorXd &delta) { this->delta = delta; }

    VectorXd get_delta() { return delta; }

    MatrixXd get_weights() { return weights; }

    void updateWeights(const VectorXd &input, double learningRate) {
        weights -= learningRate * input * delta.transpose();
        biases -= learningRate * delta;
    }
};
