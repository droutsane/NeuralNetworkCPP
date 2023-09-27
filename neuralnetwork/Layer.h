#pragma once
#include<vector>
#include "Neuron.h"

class Layer
{
public:
	Layer() {}
	~Layer() {}

	void printLayer() const{}

protected:
	std::vector<Neuron> listofNeurons;
	size_t numberofNeuronInLayer;

};
