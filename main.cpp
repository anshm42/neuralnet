#include "data.hpp"
#include "network.hpp"
#include <ctime>

const std::string mnist_train_data_path = "dataset/train-images.idx3-ubyte";
const std::string mnist_train_label_path = "dataset/train-labels.idx1-ubyte";
const std::string mnist_test_data_path = "dataset/t10k-images.idx3-ubyte";
const std::string mnist_test_label_path = "dataset/t10k-labels.idx1-ubyte";

using namespace Eigen;

int main() {
    srand(time(nullptr));

    std::vector<VectorXd> trainingData;
    std::vector<VectorXd> trainingDataLabels;

    std::vector<VectorXd> testingDataset;
    std::vector<VectorXd> testingDatasetLabels;

    read_mnist_train_data(mnist_train_data_path, trainingData);
    read_mnist_train_label(mnist_train_label_path, trainingDataLabels);
    read_mnist_test_data(mnist_test_data_path, testingDataset);
    read_mnist_test_label(mnist_test_label_path, testingDatasetLabels);

    std::cout << "Training data loaded: " << trainingData.size() << " samples"
              << std::endl;
    std::cout << "Testing data loaded: " << testingDataset.size() << " samples"
              << std::endl;

    if (trainingData.size() == 0 || testingDataset.size() == 0) {
        std::cerr << "Error: Failed to load datasets. Check file paths."
                  << std::endl;
        return 1;
    }

    Network network(784, 256, ActivationType::LEAKY_RELU);
    network.addLayer(128, ActivationType::LEAKY_RELU);
    network.addLayer(64, ActivationType::LEAKY_RELU);
    network.addLayer(32, ActivationType::LEAKY_RELU);
    network.addLayer(10, ActivationType::SOFTMAX);

    double learningRate = 0.003;
    int batchSize = 32;
    int epochs = 16;
    double decayRate = 0.95;

    network.train(trainingData, trainingDataLabels, learningRate, batchSize,
                  epochs, decayRate);

    network.test(testingDataset, testingDatasetLabels);

    return 0;
}
