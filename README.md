# Evolvable
A C++ Neuroevolution library to easily add evolvable and augmenting neural networks. Inpired by the NEAT algorithm by Kenneth Stanley, the topology or the neural network will change over time unlike classic Neuroevoltuion with a static topology meaning that you dont have to specify the topology of the network manually.

<h2>Usage</h2>

To use, simply download the ```NeuralNetwork.hpp``` file from the repo or releases page and just reference it with ```#include "NeuralNetwork.hpp"```

```c++
#include <iostream>
#include <random>
#include <math.h>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
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
    nn.mutate(0.3, true); //Mutate the neural network by a rate from 0 to 1. Boolean to specify whether or not to mutate topology
    nn = nn.crossover(nn, nn2); //Combine 2 neural networks
    nn.printNetwork(true); //true or false to print weights or not

    return 0;
}
```

<h2>XOR example</h2>

```c++
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
    vector<int> topology = {2, 1}; //The topology of the neural network, topology will change throughout evolution

    //XOR problem: if 2 values are equal then return 0, if 2 values are not equal then return 1

    vector<NeuralNetwork> nns; //population
  vector<float> fitness; //population fitnesses
  for (int i=0;i<200;i++){
    nns.push_back(NeuralNetwork(topology));
    fitness.push_back(0);
  }
  vector<float> out;
  NeuralNetwork bestEver = NeuralNetwork(topology);
  float bestEverFit = 0;
  vector<vector<int>> allIns; //All possible cobinations of XOR to remove random generation
  allIns.push_back({0, 1});
  allIns.push_back({0, 0});
  allIns.push_back({1, 0});
  allIns.push_back({1, 1});
  for (int g=0;g<1000;g++){

    //Ga Stuff
    for (int i=0;i<nns.size();i++){
      for (int x=0;x<allIns.size();x++){
          inputs = {(float)allIns[x][0], (float)allIns[x][1]};
          out = nns[i].feedforward(inputs);
            //Calculate fitness
            if (inputs[0] == inputs[1] && out[0] < 0.5){
                fitness[i] += 1-out[0];
            }
            else if (out[0] > 0.5 && inputs[0] != inputs[1]){
                fitness[i] += out[0]-0.5;
            }
            else{
                fitness[i] += -1;
            }
            
      }
      
      
    }
    int best = 0;
    float highest = 0;
    vector<int> elitists = {0, 0};
    //Get best 2 networks
    for (int i=0;i<nns.size();i++){
      if (highest < fitness[i]){
        elitists[0] = elitists[1];
        elitists[1] = i;
        highest = fitness[i];
        best = i;
      }
      fitness[i] = 0;
    }

    printf("Generation: %d\n" , g);

    NeuralNetwork BestNN = nns[elitists[0]];
    //save the best ever network
    if (highest > bestEverFit){
        bestEverFit = highest;
        bestEver = BestNN;
    }
    
    for (int i=0;i<nns.size();i++){
      //Crossover 2 best networks and mutate by 20%
      nns[i] = nns[i].crossover(nns[elitists[0]], nns[elitists[1]]);
      nns[i].mutate(0.2, true);
    }
    //---------------------------------------------------------------------
  }
  for (int i=0;i<30;i++){
    vector<float> testInput = {round(randf(0, 1)), round(randf(0, 1))};
    vector<float> output = bestEver.feedforward(testInput);
    printf("Test Case: %f %f\n", testInput[0], testInput[1]);
    
    printf("Result: %f\n\n", output[0]);
  }
    bestEver.printNetwork(true);
  

    return 0;
}
```
