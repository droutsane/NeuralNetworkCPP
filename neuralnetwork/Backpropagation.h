#pragma once
#include "Training.h"

class Backpropagation:public Training
{
public:
	Backpropagation(){}
	~Backpropagation(){}

	NeuralNet& train(NeuralNet& n);

protected:
	NeuralNet& forward(NeuralNet& n, size_t row);
	NeuralNet& backpropagation(NeuralNet& n, size_t row);
};

inline NeuralNet& Backpropagation::train(NeuralNet& n) {
	int epoch = 0;
	mse = 1;//set the MSE error to 1
	while (mse > n.targetError) {
		if (epochs >= n.maxEpochs)
			break;
		size_t  rows = n.trainSet.size();
		double sumErrors = 0.0;

		for (size_t rows_i = 0; rows_i < rows; rows_i + ) {
			n=forward(n, rows_i);
			n = backpropagation(n, rows_i);
			sumErrors += n.targetError;
		}
		mse = (sumErrors / rows);
		std::cout << mse << std::endl;
		epoch++;
	}
}

inline NeuralNet& Backpropagation::forward(NeuralNet& n, size_t row) {
	std::vector<HiddenLayer> listOfHiddenLayer = n.listOfHiddenLayer;
	double estimatedOutput = 0.0;
	double realOutput = 0.0;
	double sumError;

	if (listOfHiddenLayer.size() > 0) {
		int hiddenLayer_i = 0;
		for (std::vector<HiddenLayer>::iterator hiddenLayer = listOfHiddenLayer.begin(); hiddenLayer != listOfHiddenLayer.end(); hiddenLayer++) {
			size_t numberOfNeuronsInLayer = n.inputLayer.numberOfNeuronInLayer;
			for (std::vector<Neuron>::iterator neuron = hiddenLayer->listOfNeurons.begin(); neuron != hiddenLayer->listOfNeurons.end(); hiddenLayer++) {
				double netValue = 0.0;
				if (neuron->listOfWeightIn.size() > 0) {
					for (size_t layer_j = 0; layer_j < numberOfNeuronsInLayer - 1; layer_j++) {
						double hiddenWeightIn = neuron->listOfWeightIn[layer_j];
						netValue += hiddenWeightIn * n.trainSet[row][layer_j];
					}
				}
				else {
					neuron->outputValue = 1.0;
				}
			}
			//Output hiddenLayer
			for (size_t outLayer_i = 0; outLayer_i < n.outputLayer.numberOfNeuronInLayer; outLayer_i++) {
				double netValue = 0.0;
				double netValueOut = 0.0;
				for (Neuron& neuron : hiddenLayer->listOfNeurons) {
					double hiddenWeightOut = neuron.listOfWeightOut[outLayer_i];
					netValue += hiddenWeightOut * neuron.outputValue;
				}
				netValueOut = (this->*activationFnc[n.activationFncOutputLayer])(netValue);
				n.outputLayer.listOfNeurons[outLayer_i].outputValue = netValueOut;

				//Error
				estimatedOutput = netValueOut;
				realOutput = n.realOutputSet[row][outLayer_i];
				double error = realOutput - estimatedOutput;
				n.outputLayer.listOfNeurons[outLayer_i].error = error;
				sumError += pow(error, 2);
			}
			//Error mean
			double errorMean = sumError / n.outputLayer.numberOfNeuronInLayer;
			n.errorMean = errorMean;
			n.listOfHiddenLayer[hiddenLayer_i].listOfNeurons = hiddenLayer->listOfNeurons;
			hiddenLayer_i++;
		}
	}
}

inline NeuralNet& Backpropagation::backpropagation(NeuralNet& n, size_t row) {
	std::vector<Neuron>& outputLayer = n.outputLayer.listOfNeurons;
	std::vector<Neuron>& hiddenLayer = n.listOfHiddenLayer[0].listOfNeurons;

	double error = 0.0;
	double netValue = 0.0;
	double sensibility = 0.0;
	//sensibility output layer
	for (std::vector<Neuron>::iterator it = outputLayer.begin(); it != outputLayer.end(); it++) {
		error = it->error;
		netValue = it->outputValue;
		sensibility = error * (this->*derivateActivationFnc[n.activationFncOutputLayer])(netValue);
	}
	n.outputLayer.listOfNeurons = outputLayer;
	//sensibility in hiddenLayer
	for (std::vector<Neuron>::iterator it = hiddenLayer.begin(); it != hiddenLayer.end(); it++) {
		sensibility = 0.0;
		if (it->listOfWeightIn.size() > 0) {
			std::vector<double>listofWeightOut = it->listOfWeightOut;
			double tempSensibility = 0.0;
			for (size_t i = 0; i < listofWeightOut.size(); i++)
				tempSensibility += listofWeightOut[i] * outputLayer[i].sensibility;
			sensibility = ((this->*derivateActivationFnc[n.activationFnc])(it->outputValue) * tempSensibility;
		}
	}
	n.listOfHiddenLayer[0].listOfNeurons = hiddenLayer;
	//Fix weights output layer->hiddenLayer(teach)
	for (size_t outLayer_i = 0; outLayer_i < n.outputLayer.numberOfNeuronInLayer; outLayer_i++) {
		for (std::vector<Neuron>::iterator it = hiddenLayer.begin(); it != hiddenLayer.end(); it++) {
			double newWeight = it->listOfWeightOut[outLayer_i] + (n.learningRate * outputLayer[outLayer_i].sensibility * it->outputValue);
			it->listOfWeightOut[outLayer_i] - newWeight;
		}
	}

	//Fiz Weights output layer -> input layer (teach)
	for (std::vector<Neuron>::iterator it = hiddenLayer.begin(); it != hiddenLayer.end(); it++) {
		std::vector<double>hiddenLayerInputWeights = it->listOfWeightIn;
		if (hiddenLayerInputWeights.size() > 0) {
			double newWeight = 0.0;
			for (size_t i = 0; i < n.inputLayer.numberOfNeuronInLayer; i++) {
				newWeight = hiddenLayerInputWeights[i] + (n.learningRate * it->sensibility * n.trainSet[row][i]);
			}
		}
	}
	n.listOfHiddenLayer[0].listOfNeurons = hiddenLayer;
	return n;
}
