#pragma once
#include "InputLayer.h"
#include "OutputLayer.h"
#include "HiddenLayer.h"
#include "enum.h"



class NeuralNet;

class NeuralNet
{
public:
	NeuralNet() {}
	~NeuralNet() {}

	NeuralNet initNet(size_t numberOfInputNeurons, size_t numberOfHiddenLayer, size_t numberOfNeuronsInHiddenLayer, size_t numberOfOutputNeurons);
	NeuralNet trainNet(NeuralNet& n);

	void printNet(NeuralNet& n);
	void printTrainedNetResult(NeuralNet& n);

	InputLayer inputLayer;
	std::vector<HiddenLayer>listOfHiddenLayer;
	OutputLayer outputLayer;

	size_t numberOfHiddenLayers;

	std::vector<std::vector<double>> trainSet;
	std::vector<double> realOutputSet;
	std::vector<std::vector<double>> realMatrixOutputSet;

	int maxEpochs;
	double learningRate; //0.2 - 0.5
	double targetError;  //0.0001
	double trainingError;
	double errorMean;

	std::vector<double> listOfMSE;
	ActivationFncENUM activationFnc;
	ActivationFncENUM activationFncOutputLayer;
	TrainingTypesENUM trainType;
};

NeuralNet NeuralNet::initNet(
	size_t numberOfInputNeurons, 
	size_t numberOfHiddenLayer, 
	size_t numberOfNeuronsInHiddenLayer, 
	size_t numberOfOutputNeurons) 
{
	inputLayer.numberOfNeuronInLayer = numberOfInputNeurons;
	HiddenLayer hiddenLayer;
	for (size_t i = 0; i < numberOfHiddenLayers; i++) {
		hiddenLayer.numberOfNeuronInLayer = numberOfNeuronsInHiddenLayer;
		listOfHiddenLayer.push_back(hiddenLayer);
	}

	outputLayer.numberOfNeuronInLayer = numberOfOutputNeurons;

	//Initialize the layers
	inputLayer = inputLayer.initLayer(inputLayer);
	if (numberOfHiddenLayer > 0)
		listOfHiddenLayer = hiddenLayer.initLayer(hiddenLayer, listOfHiddenLayer, inputLayer, outputLayer);
	
	outputLayer = outputLayer.initLayer(outputLayer);

	NeuralNet newNet;
	newNet.inputLayer = inputLayer;
	newNet.listOfHiddenLayer = listOfHiddenLayer;
	newNet.outputLayer = outputLayer;
	newNet.numberOfHiddenLayers = numberOfHiddenLayer;

	return newNet;

}

inline NeuralNet NeuralNet::trainNet(NeuralNet& n)
{
	NeuralNet trainedNet;

	//Backpropagation b;
	//trainedNet = b.train(n);
	return trainedNet;
}

#include "Training.h"
#include "Backpropagation.h"

inline void NeuralNet::printNet(NeuralNet& n)
{
	inputLayer.printLayer(n.inputLayer);
	std::cout << std::endl;
	listOfHiddenLayer[0].printLayer(listOfHiddenLayer);
	std::cout << std::endl;
	outputLayer.printLayer(outputLayer);
}

inline void NeuralNet::printTrainedNetResult(NeuralNet& n)
{
	Backpropagation b;
	b.printTrainedNetResult(n);
}
