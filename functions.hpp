#include <Eigen/Dense>

using namespace Eigen;

inline VectorXd sigmoid(const VectorXd &v) {
    return 1 / (1 + (-v.array()).exp());
}

inline VectorXd dsigmoid(const VectorXd &v) {
    return sigmoid(v).array() * (1.0 - sigmoid(v).array());
}

inline double MSE(const VectorXd &expectedOut, const VectorXd &trueOut) {
    return (expectedOut.array() - trueOut.array()).square().mean();
}

inline VectorXd dMSE(const VectorXd &expectedOut, const VectorXd &trueOut) {
    return 2 * (expectedOut - trueOut) / expectedOut.size();
}
