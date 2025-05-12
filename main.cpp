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

    Network network(784, 64);
    network.addLayer(64, 8);
    network.addLayer(8, 10);

    network.train(trainingData, trainingDataLabels, 0.1, 3);
    network.test(testingDataset, testingDatasetLabels);

    return 0;
}
