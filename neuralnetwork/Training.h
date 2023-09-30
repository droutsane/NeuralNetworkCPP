#pragma once
#include<vector>
class Training
{
public:
	Training() {}
	~Training() {}

	NeuralNet& train(NeuralNet&) {}
	void printTrainedNetResult(NeuralNet& trainedNet){}


	double fncStep(double n) { return (n >= 0) ? 1.0 : 0.0; }
	double fncLinear(double n) { return n; }
	double fncSigLog(double n) { return 1.0 / (1.0 / exp(-n)); }
	double fncHyperTan(double n) { return tanh(n); }

	double derivativeFncLinear(double n) { return n; }
	double derivativeFncSigLog(double n) { return n * (1.0 - n); }
	double derivativeFncHyperTan(double n) { return 1.0 / pow(cosh(n), 2); }

public:
	typedef double(Training::*fptr)(double);
	fptr activationFnc[4] = { &Training::fncLinear, &Training::fncSigLog, &Training::fncHyperTan, &Training::fncStep };
	fptr derivateActivationFnc[3] = {&Training::derivativeFncLinear, &Training::derivativeFncSigLog, &Training::derivativeFncHyperTan};

	int epochs = 0;
	double error = 0;
	double mse = 0;
};


inline void Training::printTrainedNetResult(NeuralNet& trainedNet) {
	size_t rows = sizeof(trainedNet.trainSet) / sizeof(trainedNet.trainSet[0]);
	size_t cols = sizeof(trainedNet.trainSet[0]) / sizeof(double);

	std::vector<double>inputWeightIn;

	for (size_t i = 0; i < rows; i++) {
		double netValue = 0.0;
		for (size_t j = 0; j < cols; j++) {
			inputWeightIn = trainedNet.inputLayer.listOfNeurons[j].listOfWeightIn;
			double inputWeight = inputWeightIn[0];
			netValue += inputWeight * trainedNet.trainSet[i][j];
			

			std::cout << trainedNet.trainSet[i][j] << "\t";

		}
		double estimatedOutput =  (this->*activationFnc[trainedNet.activationFnc])(netValue);

		std::cout << " NET OUTPUT:" << estimatedOutput << "\t";
		std::cout << "REAL OUTPTU:" << trainedNet.realOutputSet[i] << "\t";
		double error = estimatedOutput - trainedNet.realOutputSet[i];
		std::cout << "ERROR" << error << std::endl;
	}

}