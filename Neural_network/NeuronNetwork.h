#include <iostream>
#include <vector>
#include <functional>
#include "Neuron.h"
#include "../DECC_G/DECC_G.h"
#include "../general/general_var.h"
#include <fstream>
#include <ctime>
using namespace std;

class NeuronNetwork
{
private:
	int layerCount;//Количество слоев или строки
	int neuronCount;//Количество нейронов в каждом слое, столбцы
	int inCount;//Количество входящих параметров

	double* outCoef = nullptr;//Выходные коэффициенты для получения ОДНОГО y
	Neuron** grid = nullptr;//Сетка из нейронов
	//Обычное умножение матрицы строки на матрицу столбец, для получения вектора суммы в нейронах, а после использование функции активации нейрона
	double* matrixMulti(Neuron*, double*, int);
	double getError(double* w, double* x, double* y, int dataSize);
	
	
	double errorCombination = 9999999;//Ошибка для сохранения комбинации
	double* bestCombination = nullptr;//Наилучшие значения для весовых коэффициентов

	void setFuncActivation(vector<double>);
	
	
	
	void ADAMTrain(double** x , double* y , int dataSize , int numEpochs , double beta1 , double beta2 , double learningRate);

public:
	void changeFromFile(string nameFile);
	double getError(double** x, double* y, int dataSize);
	NeuronNetwork(int layerCount, int neuronCount,int inCount, int numUsedFuncActivation = 0);
	void saveSettings(string fileName);
	double getValue(double*);
	void changeW(double* w);
	
	void startTrain(double** x , double* y , int dataSize , int numEpochs, double beta1, double beta2, double learningRate, int fevGlobal, int T);


	~NeuronNetwork()
	{
		if (outCoef != nullptr) {
			delete[] outCoef;
			outCoef = nullptr;
		}
		if (grid != nullptr) {
			for (int i = 0; i < layerCount; i++) {
				delete[] grid[i];
			}
			delete[] grid;
			grid = nullptr;
		}
		if (bestCombination != nullptr) {
			delete[] bestCombination;
			bestCombination = nullptr;
		}
	}
};

