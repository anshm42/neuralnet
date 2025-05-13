#include "network.hpp"
#include "data.hpp"

const std::string mnist_train_data_path = "dataset/train-images.idx3-ubyte";
const std::string mnist_train_label_path = "dataset/train-labels.idx1-ubyte";
const std::string mnist_test_data_path = "dataset/t10k-images.idx3-ubyte";
const std::string mnist_test_label_path = "dataset/t10k-labels.idx1-ubyte";

using namespace Eigen;

int main() {
    std::vector<VectorXd> trainingData;
    std::vector<VectorXd> trainingDataLabels;

    std::vector<VectorXd> testingDataset;
    std::vector<VectorXd> testingDatasetLabels;

    read_mnist_train_data(mnist_train_data_path, trainingData);
    read_mnist_train_label(mnist_train_label_path, trainingDataLabels);
    read_mnist_test_data(mnist_test_data_path, testingDataset);
    read_mnist_test_label(mnist_test_label_path, testingDatasetLabels);

    Network network(784, 512, ActivationType::RELU);
    
    network.addLayer(512, 256, ActivationType::LEAKY_RELU);
    network.addLayer(256, 128, ActivationType::RELU);
    network.addLayer(128, 64, ActivationType::LEAKY_RELU);
    network.addLayer(64, 32, ActivationType::SIGMOID);
    
    network.addLayer(32, 10, ActivationType::SIGMOID);
    
    double learningRate = 0.001;
    int batchSize = 128;
    int epochs = 25;
    double decayRate = 0.9;
    
    network.train(trainingData, trainingDataLabels, learningRate, batchSize, epochs, decayRate);
    
    network.test(testingDataset, testingDatasetLabels);

    return 0;
}
