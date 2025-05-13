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
        if (actType == ActivationType::RELU || actType == ActivationType::LEAKY_RELU) {
            weights = MatrixXd::Random(in, out) * sqrt(2.0 / in);
        } else {
            weights = MatrixXd::Random(in, out) * sqrt(2.0 / (in + out));
        }
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
            case ActivationType::SOFTMAX:
                activations = softmax(y);
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
            case ActivationType::SOFTMAX:
                return dSoftmax(m);
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
        double lambda = 0.0001;
        
        MatrixXd dW = batchInput * delta.transpose() / batchInput.cols();
        VectorXd db = delta.rowwise().mean();
        
        dW += lambda * weights;
        
        double clipThreshold = 5.0;
        
        for (int i = 0; i < dW.rows(); i++) {
            for (int j = 0; j < dW.cols(); j++) {
                if (dW(i, j) > clipThreshold) dW(i, j) = clipThreshold;
                if (dW(i, j) < -clipThreshold) dW(i, j) = -clipThreshold;
                
                if (std::isnan(dW(i, j))) {
                    dW(i, j) = 0.0;
                }
            }
        }
        
        for (int i = 0; i < db.size(); i++) {
            if (db(i) > clipThreshold) db(i) = clipThreshold;
            if (db(i) < -clipThreshold) db(i) = -clipThreshold;
            
            if (std::isnan(db(i))) {
                db(i) = 0.0;
            }
        }
        
        weights -= lr * dW;
        biases -= lr * db;
    }
};