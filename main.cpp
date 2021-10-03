#include <iostream>
#include <random>
#include <math.h>
#include <vector>
#include <stdio.h>      /* printf, NULL */
#include <stdlib.h>     /* srand, rand */
#include <time.h>  
#include <future>
#include <vector>
#include "NeuralNetwork.hpp"

using namespace std;

int main(){
    vector<int> topology = {2, 3, 4}; //The topology of the neural network (2 inputs, 3 hidden neurons, 4 outputs)
    vector<float> inputs = {1, 5};
    vector<float> output;
    NeuralNetwork nn = NeuralNetwork(topology);
    NeuralNetwork nn2 = NeuralNetwork(topology);
    output = nn.feedforward(inputs); //Generate an output from some input
    nn.mutate(0.3); //Mutate the neural network by a rate from 0 to 1
    nn = nn.crossover(nn, nn2); //Combine 2 neural networks
    nn.printNetwork(true); //true or false to print weights or not

    return 0;
}