#include <Eigen/Dense>

using namespace Eigen;

enum class ActivationType {
    RELU,
    SIGMOID,
    LEAKY_RELU
};

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

inline MatrixXd sigmoid(const MatrixXd &m) {
    return 1 / (1 + (-m.array()).exp());
}

inline MatrixXd dSigmoid(const MatrixXd &m) {
    return sigmoid(m).array() * (1.0 - sigmoid(m).array());
}

inline double MSE(const MatrixXd& target, const MatrixXd& real) {
    return (target - real).array().square().mean();
}

inline MatrixXd dMSE(const MatrixXd& target, const MatrixXd& real) {
    return 2 * (target - real) / target.cols();
}

inline MatrixXd relu(const MatrixXd &m) {
    return m.array().max(0);
}

inline MatrixXd dRelu(const MatrixXd &m) {
    return (m.array() > 0).cast<double>();
}

inline MatrixXd leakyRelu(const MatrixXd &m, double alpha = 0.01) {
    return m.array().max(alpha * m.array());
}

inline MatrixXd dLeakyRelu(const MatrixXd &m, double alpha = 0.01) {
    return (m.array() > 0).select(MatrixXd::Ones(m.rows(), m.cols()), MatrixXd::Constant(m.rows(), m.cols(), alpha));
}
