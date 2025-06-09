#include "layer.hpp"
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

class Network {
  private:
    std::vector<Layer> layers;

  public:
    Network(int in, int hidden, ActivationType actType = ActivationType::RELU) {
        layers.emplace_back(in, hidden, actType);
    }

    void addLayer(int neurons, ActivationType actType = ActivationType::RELU) {
        int in = layers.back().getWeights().cols();
        layers.emplace_back(in, neurons, actType);
    }

    void forward(const MatrixXd &batchInput) {
        layers[0].forward(batchInput);
        for (size_t i = 1; i < layers.size(); i++) {
            layers[i].forward(layers[i - 1].getActivations());
        }
    }

    void backward(const MatrixXd &batchInput, const MatrixXd &batchTarget,
                  double lr) {
        MatrixXd output = layers.back().getActivations();

        MatrixXd delta;
        if (layers.back().getActivationType() == ActivationType::SOFTMAX) {
            delta = output - batchTarget;
        } else {
            MatrixXd error = output - batchTarget;
            delta = error.cwiseProduct(
                layers.back().getActivationDerivative(output));
        }

        layers.back().setDelta(delta);

        for (int i = layers.size() - 2; i >= 0; i--) {
            MatrixXd nextWeights = layers[i + 1].getWeights();
            MatrixXd nextDelta = layers[i + 1].getDelta();

            MatrixXd hiddenError = nextWeights * nextDelta;

            MatrixXd hiddenOutput = layers[i].getActivations();
            MatrixXd hiddenDelta = hiddenError.cwiseProduct(
                layers[i].getActivationDerivative(hiddenOutput));

            layers[i].setDelta(hiddenDelta);
        }

        for (size_t i = 0; i < layers.size(); i++) {
            MatrixXd layerInput;
            if (i == 0) {
                layerInput = batchInput;
            } else {
                layerInput = layers[i - 1].getActivations();
            }
            layers[i].updateWeights(layerInput, lr);
        }
    }

    void train(const std::vector<VectorXd> &data,
               const std::vector<VectorXd> &labels, double learningRate,
               int batchSize, int epochs = 20, double decayRate = 0.8) {
        int numSamples = data.size();
        int numBatches = numSamples / batchSize;

        std::vector<int> indices(numSamples);
        for (int i = 0; i < numSamples; i++) {
            indices[i] = i;
        }

        std::cout << "\n===== NEURAL NETWORK TRAINING STARTED =====\n";
        std::cout << "Total samples: " << numSamples << "\n";
        std::cout << "Batch size: " << batchSize << "\n";
        std::cout << "Batches per epoch: " << numBatches << "\n";
        std::cout << "Initial learning rate: " << learningRate << "\n";
        std::cout << "Learning rate decay: " << decayRate
                  << " (every 5 epochs)\n";
        std::cout << "Total epochs: " << epochs << "\n";
        std::cout << "Network architecture: ";
        for (size_t i = 0; i < layers.size(); i++) {
            std::string actType;
            switch (layers[i].getActivationType()) {
            case ActivationType::SIGMOID:
                actType = "SIGMOID";
                break;
            case ActivationType::LEAKY_RELU:
                actType = "LEAKY_RELU";
                break;
            case ActivationType::RELU:
                actType = "RELU";
                break;
            case ActivationType::SOFTMAX:
                actType = "SOFTMAX";
                break;
            default:
                actType = "UNKNOWN";
                break;
            }
            if (i < layers.size() - 1) {
                std::cout << layers[i].getWeights().rows() << " (" << actType
                          << ") → ";
            } else {
                std::cout << layers[i].getWeights().rows() << " (" << actType
                          << ") → " << layers[i].getWeights().cols() << "\n";
            }
        }
        std::cout << "=======================================\n\n";

        std::random_device rd;
        std::mt19937 g(rd());

        double lr = learningRate;
        double bestLoss = std::numeric_limits<double>::max();
        double bestAccuracy = 0.0;

        for (int epoch = 0; epoch < epochs; epoch++) {
            if (epoch > 0 && epoch % 5 == 0) {
                lr *= decayRate;
                std::cout << "Learning rate reduced to: " << std::fixed
                          << std::setprecision(6) << lr << "\n";
            }

            std::cout << "Epoch " << (epoch + 1) << "/" << epochs
                      << " started\n";
            std::shuffle(indices.begin(), indices.end(), g);

            double totalLoss = 0.0;
            int progressStep = numBatches / 50;
            if (progressStep == 0)
                progressStep = 1;

            for (int batch = 0; batch < numBatches; batch++) {
                MatrixXd batchInput(data[0].size(), batchSize);
                MatrixXd batchTarget(labels[0].size(), batchSize);

                for (int i = 0; i < batchSize; i++) {
                    int idx = indices[batch * batchSize + i];
                    batchInput.col(i) = data[idx];
                    batchTarget.col(i) = labels[idx];
                }

                forward(batchInput);

                MatrixXd output = layers.back().getActivations();
                totalLoss += MSE(batchTarget, output);

                backward(batchInput, batchTarget, lr);

                if (batch % progressStep == 0 || batch == numBatches - 1) {
                    int progress = (batch + 1) * 100 / numBatches;
                    std::cout << "\rProgress: [";
                    int barWidth = 30;
                    int pos = barWidth * progress / 100;
                    for (int i = 0; i < barWidth; ++i) {
                        if (i < pos)
                            std::cout << "=";
                        else if (i == pos)
                            std::cout << ">";
                        else
                            std::cout << " ";
                    }
                    std::cout << "] " << progress << "% (" << (batch + 1) << "/"
                              << numBatches << " batches)";
                    std::cout.flush();
                }
            }

            totalLoss /= numBatches;
            std::cout << "\nEpoch " << (epoch + 1) << "/" << epochs
                      << " completed. Avg Loss: " << std::fixed
                      << std::setprecision(6) << totalLoss << "\n";

            if (totalLoss < bestLoss) {
                bestLoss = totalLoss;
                std::cout << "New best loss: " << std::fixed
                          << std::setprecision(6) << bestLoss << "\n";
            }

            if ((epoch + 1) % 2 == 0 || epoch == epochs - 1) {
                std::vector<VectorXd> validationData;
                std::vector<VectorXd> validationLabels;

                int validationSize = std::min(5000, (int)data.size());
                validationData.reserve(validationSize);
                validationLabels.reserve(validationSize);

                for (int i = 0; i < validationSize; i++) {
                    int idx = indices[i];
                    validationData.push_back(data[idx]);
                    validationLabels.push_back(labels[idx]);
                }

                double validationAccuracy =
                    test(validationData, validationLabels, true);
                std::cout << "Validation Accuracy: " << std::fixed
                          << std::setprecision(2) << validationAccuracy
                          << "%\n";

                if (validationAccuracy > bestAccuracy) {
                    bestAccuracy = validationAccuracy;
                    std::cout << "New best validation accuracy: " << std::fixed
                              << std::setprecision(2) << bestAccuracy << "%\n";
                }
            }

            std::cout << "---------------------------------------\n";
        }

        std::cout << "\n===== TRAINING COMPLETED =====\n";
        std::cout << "Final best loss: " << std::fixed << std::setprecision(6)
                  << bestLoss << "\n";
        std::cout << "Best validation accuracy: " << std::fixed
                  << std::setprecision(2) << bestAccuracy << "%\n";
        std::cout << "==============================\n\n";
    }

    double test(const std::vector<VectorXd> &data,
                const std::vector<VectorXd> &labels,
                bool isValidation = false) {
        int correct = 0;
        double totalLoss = 0.0;

        if (!isValidation) {
            std::cout << "\n===== NEURAL NETWORK TESTING =====\n";
            std::cout << "Testing on " << data.size() << " samples\n";
            std::cout << "=================================\n\n";
        }

        int progressStep = data.size() / 100;
        if (progressStep == 0)
            progressStep = 1;

        for (size_t i = 0; i < data.size(); i++) {
            MatrixXd input(data[i].size(), 1);
            input.col(0) = data[i];

            forward(input);

            VectorXd output = layers.back().getActivations().col(0);

            MatrixXd target(labels[i].size(), 1);
            target.col(0) = labels[i];
            totalLoss += MSE(target, layers.back().getActivations());

            int predicted = 0;
            double maxVal = output(0);
            for (int j = 1; j < output.size(); j++) {
                if (output(j) > maxVal) {
                    maxVal = output(j);
                    predicted = j;
                }
            }

            int actual = 0;
            for (int j = 0; j < labels[i].size(); j++) {
                if (labels[i](j) > 0.5) {
                    actual = j;
                    break;
                }
            }

            if (predicted == actual) {
                correct++;
            }

            if (!isValidation &&
                (i % progressStep == 0 || i == data.size() - 1)) {
                int progress = (i + 1) * 100 / data.size();
                std::cout << "\rProgress: [";
                int barWidth = 30;
                int pos = barWidth * progress / 100;
                for (int j = 0; j < barWidth; ++j) {
                    if (j < pos)
                        std::cout << "=";
                    else if (j == pos)
                        std::cout << ">";
                    else
                        std::cout << " ";
                }
                std::cout << "] " << progress << "% (" << (i + 1) << "/"
                          << data.size() << " samples)";
                std::cout.flush();
            }
        }

        double accuracy = (double)correct / data.size() * 100.0;
        double avgLoss = totalLoss / data.size();

        if (!isValidation) {
            std::cout << "\n\n===== TESTING COMPLETED =====\n";
            std::cout << "Test set average loss: " << std::fixed
                      << std::setprecision(6) << avgLoss << "\n";
            std::cout << "Test set accuracy: " << std::fixed
                      << std::setprecision(2) << accuracy << "% (" << correct
                      << "/" << data.size() << " correct)\n";
            std::cout << "============================\n\n";
        }

        return accuracy;
    }
};