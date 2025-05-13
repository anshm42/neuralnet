#include <Eigen/Dense>

using namespace Eigen;

enum class ActivationType {
    RELU,
    SIGMOID,
    LEAKY_RELU,
    SOFTMAX
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

inline MatrixXd softmax(const MatrixXd &m) {
    MatrixXd result(m.rows(), m.cols());
    for (int i = 0; i < m.cols(); i++) {
        VectorXd col = m.col(i);
        double max_val = col.maxCoeff();
        col = col.array() - max_val; 
        col = col.array().exp();
        col = col / col.sum();
        result.col(i) = col;
    }
    return result;
}

inline MatrixXd dSoftmax(const MatrixXd &m) {
    return MatrixXd::Ones(m.rows(), m.cols());
}
