/*
Evolvable
Inspired by Kenneth Stanley's NEAT algorithm
Written by Anik Patel 2021
*/
#include <iostream>
#include <random>
#include <math.h>
#include <vector>
#include <stdio.h>      /* printf, NULL */
#include <stdlib.h>     /* srand, rand */
#include <time.h>  
#include <future>
#include <vector>
#include <fstream>

#define E 2.71828182845904

using namespace std;



float randf(float lo, float hi) {
    float random = ((float) rand()) / (float) RAND_MAX;
    float diff = hi - lo;
    float r = random * diff;
    return lo + r;
}
float sigmoid(float x) {
  if (x == 0){
    return 0;
  }
  if (x == 1){
    //return 1;
  }
  if (x < 0){
    return 0;
  }
  float y = 1 / (1 + pow((float)E, -4.9*x));
  return y;
}


class Connection{
  public:
    float weight;
    int fromLayer;
    int fromNeuron;
    int toLayer;
    int toNeuron;
    float minMax = 1;
    float bias = 0;
  Connection(int tl, int tn, int fl, int fn, float w){
    fromLayer = fl;
    fromNeuron = fn;
    toLayer = tl;
    toNeuron = tn;
    weight = w;
  }
  void mutateWeight(){
    bias+=randf(-0.01, 0.01);
    if(bias > minMax){
        bias = minMax;
    }
    if(bias < -minMax){
        bias = -minMax;        
    }
    
    if (randf(0, 1) < 0.1){
      weight = randf(-minMax, minMax);
    }
    else{
      weight += randf(-0.02, 0.02);
    }
    if(weight > minMax){
        weight = minMax;
    }
    if(weight < -minMax){
        weight = -minMax;        
      }
    }
    
};

class Neuron{
  public:
    vector<Connection> connections;
    float sum;
  void mutate(float rate){
    for (int i=0;i<connections.size();i++){
      if (randf(0, 1) < rate){
        connections[i].mutateWeight();
      }
    }
  }
  
};

class Layer{
  public:
    vector<Neuron> neurons;
  Layer(int neuronCount){
    for (int i=0;i<neuronCount;i++){
      neurons.push_back(Neuron());
    }
  }
  void sig(){
    for (int i=0;i<neurons.size();i++){
      neurons[i].sum = sigmoid(neurons[i].sum);
    }
  }    
  
};

class NeuralNetwork{
  public:
    vector<Layer> layers;
    float width = 250;
    float height = 50;
    float biasNum = 0;
    int nextNode = 0;
    float rad = 10;

    

  NeuralNetwork(vector<int> topology){
    //font.loadFromFile("ARIAL.TTF");
    //text.setFont(font);
    for (int i=0;i<topology.size();i++){
      layers.push_back(Layer(topology[i]));
    }
    fullyConnect();
  }
  void printNetwork(bool cons){
    for (int i=0;i<layers.size();i++){
      for (int j=0;j<layers[i].neurons.size();j++){
        printf("%f ", layers[i].neurons[j].sum);
      }
      printf("\n");
    }
    printf("\n\n");
    if (cons){
      // for (int i=0;i<connections.size();i++){
      //   printf("FromL:%d FromN:%d ToL:%d ToN:%d Weight:%f\n", connections[i].fromLayer, connections[i].fromNeuron, connections[i].toLayer,connections[i].toNeuron, connections[i].weight);
      // }
      for (int i=0;i<layers.size();i++){
        for (int j=0;j<layers[i].neurons.size();j++){
          for (int x=0;x<layers[i].neurons[j].connections.size();x++){
            Connection c = layers[i].neurons[j].connections[x];
            printf("From Layer:%d From Neuron:%d To Layer:%d To Neuron:%d Weight:%f\n", c.fromLayer, c.fromNeuron, c.toLayer,c.toNeuron, c.weight);

          }
        }
        printf("\n");
      }

    }
    
  }
  void fullyConnect(){
    for (int i=0;i<layers.size()-1;i++){
      for (int j=0;j<layers[i].neurons.size();j++){
        for (int x=0;x<layers[i+1].neurons.size();x++){
          layers[i].neurons[j].connections.push_back(Connection(i+1, x, i, j, (randf(-1, 1))));
        }
      }
    }
    

  }
  void resetNeurons(){
    for (int i=0;i<layers.size();i++){
      for (int j=0;j<layers[i].neurons.size();j++){
        layers[i].neurons[j].sum = 0; 
      }
    }
  }
  
  void sigmoidLayer(int layer){
    //printf("%d\n", layer);
    layers[layer].sig();       
  }

  void stepLayer(int layer1){
    int toN;
    float weight;
    vector<Connection> c1;
    vector<Connection> c2;
    int layer2 = 0;
    for (int i=0;i<layers[layer1].neurons.size();i++){
      c1 = layers[layer1].neurons[i].connections;
      for (int j=0;j<c1.size();j++){
        layer2 = c1[j].toLayer;

        layers[layer2].neurons[c1[j].toNeuron].sum += layers[layer1].neurons[i].sum * c1[j].weight + c1[j].bias;
        //printf("%d \n", layer2); 
      }
    }
  }
  
  vector<float> feedforward(vector<float> ins){
    for (int i=0;i<ins.size();i++){
      layers[0].neurons[i].sum = ins[i];
    }
    vector<float> output;
    
    for (int i=0;i<layers.size()-1;i++){
      stepLayer(i);
      sigmoidLayer(i+1);
    }

    //sigmoidLayer(layers.size()-1);
    for (int i=0;i<layers[layers.size()-1].neurons.size();i++){
      output.push_back(layers[layers.size()-1].neurons[i].sum);
    }
    //printNetwork(false);
    resetNeurons();
    return output;
  }

  
  void addNode(){
    if (randf(0, 1) < 0.5){     
      int randomLayerPrev = floor(randf(0, layers.size()-2));

      for (int i=layers.size()-2;i>=randomLayerPrev+1;i--){
        //layers.erase(layers.begin()+i);
      }
      layers.insert(layers.begin()+randomLayerPrev+1, Layer(1));
        
      for (int i=randomLayerPrev+2;i<layers.size();i++){
        for (int j=0;j<layers[i].neurons.size();j++){
          for (int x=0;x<layers[i].neurons[j].connections.size();x++){
            layers[i].neurons[j].connections[x].fromLayer++;
            layers[i].neurons[j].connections[x].toLayer++;
            
          }
        }
      }
      for (int i=0;i<layers[randomLayerPrev].neurons.size();i++){
        layers[randomLayerPrev].neurons[i].connections.clear();
        layers[randomLayerPrev].neurons[i].connections.push_back(Connection(randomLayerPrev+1, 0, randomLayerPrev, i, randf(-1, 1)));
      }
      for (int i=0;i<layers[randomLayerPrev+2].neurons.size();i++){
        layers[randomLayerPrev+1].neurons[0].connections.push_back(Connection(randomLayerPrev+2, i, randomLayerPrev+1, 0, randf(-1, 1)));
      }
    
    }
    else{
      if (layers.size() > 2){
        int randomLayer = floor(randf(1, layers.size()-2));
        int neuronIndex;
        layers[randomLayer].neurons.push_back(Neuron());
        neuronIndex = layers[randomLayer].neurons.size()-1;
        for (int i=0;i<layers[randomLayer-1].neurons.size();i++){
          layers[randomLayer-1].neurons[i].connections.push_back(Connection(randomLayer, neuronIndex, randomLayer-1, i, randf(-1, 1)));

        }
        for (int i=0;i<layers[randomLayer+1].neurons.size();i++){
          layers[randomLayer].neurons[neuronIndex].connections.push_back(Connection(randomLayer+1, i, randomLayer, neuronIndex, randf(-1, 1)));
          
        }
      }
    }
    
    
    //printf("%zu\n", layers.size());
  }
  
  

  void mutate(float rate, bool changeTop){
    biasNum += randf(-0.01, 0.01);
    if (randf(0, 1) < 0.01 && changeTop){
      addNode();
    }
   
    for (int i=0;i<layers.size()-1;i++){
      for (int j=0;j<layers[i].neurons.size();j++){
        layers[i].neurons[j].mutate(rate); 
      }
    }
    
    
    // for (int i=1;i<layers.size();i++){
    //   for (int j=0;j<layers[i].neurons.size();j++){
    //     if (randf(0, 1) < 0.1){
    //         layers[0].neurons[biasNum].connections.push_back(Connection(i, j, 0, biasNum, randf(-1, 1)));
    //     }
    //   }
    // }
    
  }
  NeuralNetwork crossover(NeuralNetwork parent, NeuralNetwork parent2){
    bool cross = true;
    //parent.printNetwork(false);
    //parent2.printNetwork(false);
    
    if (parent.layers.size() == parent2.layers.size()){
      for (int i=0;i<parent.layers.size();i++){
        
        if (parent.layers[i].neurons.size() == parent2.layers[i].neurons.size()){

        }
        else{
          //printf("EEEE\n");
          cross = false;
          if (randf(0, 1) < 0.5){
            return parent2;
            
          }
          else{
            return parent;
          }
        }
      }
    }
    else{
      cross = false;
      if (randf(0, 1) < 0.5){
            return parent2;
          }
          else{
            return parent;
          }
    }
    if (cross){
      NeuralNetwork child = parent;
    for (int i=0;i<parent.layers.size()-1;i++){
      for (int j=0;j<parent.layers[i].neurons.size();j++){
        vector<Connection> con = parent2.layers[i].neurons[j].connections;
        for (int x=0;x<con.size();x++){
          if (randf(0, 1) < 0.5){
            child.layers[i].neurons[j].connections[x] = con[x];
          }
        }
      }
    }
    return child;
    }
    else{
      
    }
    if (randf(0, 1) < 0.5){
        return parent2;
      }
      else{
        return parent;
      }
    
    return parent;
    
  }
  

  

  
};


