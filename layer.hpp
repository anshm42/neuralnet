#include "functions.hpp"

class Layer {
private:
    MatrixXd weights;
    VectorXd biases;
    VectorXd values;
    VectorXd activations;

    VectorXd delta;

public:
    Layer(int in, int out) {
        weights = MatrixXd::Random(in, out);
        biases = VectorXd::Zero(out);
    }

    void forward(const VectorXd& input) {
        VectorXd y = weights.transpose() * input + biases;
        values = y;
        activations = sigmoid(y);
    }

    VectorXd getActivations() {
        return activations;
    }
    
    VectorXd dActivations() {
        return dSigmoid(values);
    }


    void setDelta(const VectorXd& delta) {
        this->delta = delta;
    }

    VectorXd getDelta() {
        return delta;
    }

    MatrixXd getWeights() {
        return weights;
    }

    void updateWeights(const VectorXd& input, double lr) {
        weights -= lr * input * delta.transpose();
        biases -= lr * delta;
    }
   
};