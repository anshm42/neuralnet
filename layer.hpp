#include "functions.hpp"

class Layer {
private:
    MatrixXd weights;
    VectorXd biases;
    MatrixXd values;
    MatrixXd activations;
    MatrixXd delta;
    
    ActivationType activationType = ActivationType::RELU;
    double leakyReluAlpha = 0.01;

public:
    Layer(int in, int out, ActivationType actType = ActivationType::RELU) 
        : activationType(actType) {
        weights = MatrixXd::Random(in, out) * sqrt(2.0 / (in + out));
        biases = VectorXd::Zero(out);
    }
    
    void setActivationType(ActivationType actType) {
        activationType = actType;
    }
    
    void setLeakyReluAlpha(double alpha) {
        leakyReluAlpha = alpha;
    }
    
    ActivationType getActivationType() const {
        return activationType;
    }

    void forward(const MatrixXd& batchInput) {
        MatrixXd y = weights.transpose() * batchInput;
        for (int i = 0; i < batchInput.cols(); i++) {
            y.col(i) += biases;
        }
        values = y;
        
        switch(activationType) {
            case ActivationType::SIGMOID:
                activations = sigmoid(y);
                break;
            case ActivationType::LEAKY_RELU:
                activations = leakyRelu(y, leakyReluAlpha);
                break;
            case ActivationType::RELU:
            default:
                activations = relu(y);
                break;
        }
    }

    void forward(const VectorXd& input) {
        MatrixXd inputMat = input;
        forward(inputMat);
    }
    
    MatrixXd getActivationDerivative(const MatrixXd& m) const {
        switch(activationType) {
            case ActivationType::SIGMOID:
                return dSigmoid(m);
            case ActivationType::LEAKY_RELU:
                return dLeakyRelu(m, leakyReluAlpha);
            case ActivationType::RELU:
            default:
                return dRelu(m);
        }
    }

    MatrixXd getActivations() const {
        return activations;
    }

    VectorXd getActivationVector() const {
        if (activations.cols() > 0) {
            return activations.col(0);
        }
        return VectorXd::Zero(activations.rows());
    }

    void setDelta(const MatrixXd& d) {
        delta = d;
    }

    MatrixXd getDelta() const {
        return delta;
    }

    MatrixXd getWeights() const {
        return weights;
    }

    void updateWeights(const MatrixXd& batchInput, double lr) {
        double lambda = 0.00001;
        
        MatrixXd dW = batchInput * delta.transpose() / batchInput.cols() + lambda * weights;
        VectorXd db = delta.rowwise().mean();
        
        weights -= lr * dW;
        biases -= lr * db;
    }
};