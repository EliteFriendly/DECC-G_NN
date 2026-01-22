#include "Neuron.h"



Neuron::Neuron(function<double(double)> funcActivation, int ammountNeuronLayer):funcActivation(funcActivation), sizeW(ammountNeuronLayer + 1)
{
	w = new double[sizeW];
	for (int i = 0; i < sizeW; i++) {
		w[i] = gen() % 1000 * 0.01;
	}
}


Neuron::Neuron(const Neuron& copy):w(copy.w),funcActivation(copy.funcActivation){}



void Neuron::changeW(double* wChange, int& counter)
{
	for (int i = 0; i < sizeW; i++) {
		w[i] = wChange[counter];
		counter++;
	}
}

double* Neuron::getW()
{
	return w;
}
