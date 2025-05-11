#include <Eigen/Dense>

using namespace Eigen;

class Layer {
  private:
    VectorXd output;

    MatrixXd weights;
    VectorXd biases;

  public:
    Layer(int input_size, int output_size) {
        weights = MatrixXd::Random(input_size, output_size);
        biases = VectorXd::Random(output_size);
    }

    VectorXd sigmoid(const VectorXd &v) {
        return 1.0 / (1.0 + (-v.array()).exp());
    }

    VectorXd sigmoid_derivative(const VectorXd &v) {
        return sigmoid(v).array() * (1.0 - sigmoid(v).array());
    }

    VectorXd forward(const VectorXd &input) {
        output = sigmoid((input * weights) + biases);
        return output;
    }
};
