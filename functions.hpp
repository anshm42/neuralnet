#include <Eigen/Dense>

using namespace Eigen;

inline VectorXd sigmoid(const VectorXd &v) {
    return 1 / (1 + (-v.array()).exp());
}

inline VectorXd dSigmoid(const VectorXd &v) {
    return sigmoid(v).array() * (1.0 - sigmoid(v).array());
}

inline double MSE(const VectorXd& target, const VectorXd& real) {
    return (target.array() - real.array()).square().mean();
}

inline VectorXd dMSE(const VectorXd& target, const VectorXd& real) {
    return 2 * (target - real) / target.size();
}
