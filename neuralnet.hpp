#include "layer.hpp"
#include <vector>

using namespace Eigen;

class NeuralNet {
  private:
    std::vector<Layer> layers;

  public:
    VectorXd forward(const VectorXd &x) {
        Vector out = x;
        for (auto L : layers)
            out = L.forward(out);
        return out;
    }

    void backward(const VectorXd &grad_output, float lr) {
        VectorXd grad = grad_output;
        for (auto it = layers.rbegin(); it != layers.rend(); ++it)
            grad = (*it).backward(grad, lr);
    }
};
