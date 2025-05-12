#include "layer.hpp"
#include <iostream>
#include <numeric>
#include <random>
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

    void train(const std::vector<Eigen::VectorXd> &trainImages,
               const std::vector<Eigen::VectorXd> &trainLabels, int epochs,
               double lr) {
        size_t N = trainImages.size();
        std::vector<size_t> idx(N);
        std::iota(idx.begin(), idx.end(), 0);
        std::mt19937_64 rng{std::random_device{}()};

        for (int e = 1; e <= epochs; ++e) {
            std::shuffle(idx.begin(), idx.end(), rng);
            double sumLoss = 0.0;

            for (size_t i : idx) {
                const auto &x = trainImages[i];
                const auto &y = trainLabels[i];

                // 1) forward
                Eigen::VectorXd y_pred = forward(x);

                // 2) compute loss + gradient
                sumLoss += MSE(y_pred, y);
                Eigen::VectorXd grad = dMSE(y_pred, y);

                // 3) backward + parameter update
                backward(grad, lr);
            }

            std::cout << "Epoch " << e << " | Train Loss: " << (sumLoss / N)
                      << "\n";
        }
    }

    std::pair<double, double>
    evaluate(const std::vector<VectorXd> &testImages,
             const std::vector<VectorXd> &testLabels) {
        size_t N = testImages.size();
        double sumLoss = 0.0;
        size_t correct = 0;

        for (size_t i = 0; i < N; ++i) {
            const auto &x = testImages[i];
            const auto &y = testLabels[i];

            VectorXd y_pred = forward(x);
            sumLoss += MSE(y_pred, y);

            Eigen::Index p, t;
            y_pred.maxCoeff(&p);
            y.maxCoeff(&t);
            if (p == t)
                ++correct;
        }

        double avgLoss = sumLoss / N;
        double accuracy = 100.0 * double(correct) / N;
        return {avgLoss, accuracy};
    }
};
