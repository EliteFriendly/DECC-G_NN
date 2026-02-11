#include <iostream>
#include <vector>
#include <functional>
#include "../general/general_var.h"

using namespace std;

class Neuron
{
private:
	double* w;//Весовые коэффициенты входящих нейронов
	function <double(double)> funcActivation;//Функция активации
	int sizeW;//Размерность вектора весов
public:
	Neuron() {}
	Neuron(const Neuron&);
	Neuron(function<double(double)>, int);

	double useFuncActivation(double x) { return funcActivation(x); }

	void changeW(double* w, int& counter);
	void replaceFuncActivation(function<double(double)> funcActivationVoid) {funcActivation = funcActivationVoid;}
	double* getW();
};

