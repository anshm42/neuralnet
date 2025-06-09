# neuralnet

Very simple neural network made in C++. Identifies hand drawn numbers from the MNIST Dataset.

## Why?

I always found machine learning to be a very confusing topic, and I wanted to learn more about it. Upon realizing the amount of linear algebra involved, I decided to create a very simple neural network to learn more about how they work.

## How to Run:

    $ g++ -O3 main.cpp -o neuralnet && ./neuralnet

## How It's Made:

**Tech used:** C++, Eigen (Linear Algebra Library)

## Example
``
    // main.cpp
    Network network(784, 256, ActivationType::LEAKY_RELU);
    network.addLayer(128, ActivationType::LEAKY_RELU);
    network.addLayer(64, ActivationType::LEAKY_RELU);
    network.addLayer(32, ActivationType::LEAKY_RELU);
    network.addLayer(10, ActivationType::SOFTMAX);

    double learningRate = 0.003;
    int batchSize = 32;
    int epochs = 16;
    double decayRate = 0.95;

### Results (On MNIST testing dataset, 10000 samples)
![Results](https://github.com/user-attachments/assets/9963197a-c2a9-4023-8f6e-75dd066a7a49)
