#include "NeuronNetwork.h"


double* NeuronNetwork::matrixMulti(Neuron* neuron, double* inMatrix, int inSize)
{
	double* out = new double[neuronCount];
	double sum = 0;//Сумма что в нейроне
	//Основной цикл перемножения слоя на входы
	for (int i = 0; i < neuronCount; i++) {
		sum = 0;
		for (int j = 0; j < inSize; j++) {
			sum += neuron[i].getW()[j] * inMatrix[j];
		}
		sum += neuron[i].getW()[inSize];//Добавление смещения нейрона
		out[i] = neuron[i].useFuncActivation(sum);
	}
	
	return out;
}



double NeuronNetwork::getError(double* w, double* x, double* y, int dataSize)
{
	changeW(w);

	double sum = 0;//Среднеквадратичная ошибка
	for (int i = 0; i < dataSize; i++) {

		sum += pow(getValue(&x[i]) - y[i], 2);
	}
	sum = pow(sum, 0.5);
	return sum;
}



void NeuronNetwork::changeW(double* w)
{
	int counter = 0;

	//Замена весовых коэффициентов слоев

	for (int i = 0; i < layerCount; i++) {
		for (int j = 0; j < neuronCount; j++) {
			grid[i][j].changeW(w, counter);

		}
	}

	//Замена выходных коэффициентов
	for (int i = 0; i < neuronCount + 1; i++) {
	    outCoef[i] = w[counter];
	    counter++;
	}
}

void NeuronNetwork::startTrain(double** x , double* y , int dataSize , int numEpochs , double beta1 , double beta2 , double learningRate,int fevGlobal, int T)
{
	int amountW = (neuronCount * (inCount + 1) + (layerCount - 1) * (neuronCount + 1) * neuronCount + (neuronCount + 1));
	double* limitsDimension = new double[amountW * 2];
	for (int i = 0; i < amountW * 2; i++) {
		if (i % 2 == 0) {
			limitsDimension[i] = -10.0;
		}
		else {
			limitsDimension[i] = 10.0;
		}
	}
	//find optimal ampount components
	int optimaullAmountComponents = 2;
	for (int numComponents = amountW; numComponents >= 2; numComponents--) {
		if (amountW % numComponents == 0 && ((amountW / numComponents) > 3 || amountW <= 21)) {
			optimaullAmountComponents = numComponents;
			break;
		}
	}
	//cout << "Ammount components:\t" << optimaullAmountComponents << endl;
	Decc_g decc_g(optimaullAmountComponents , amountW , [&](double* w) { changeW(w); return getError(x , y , dataSize); } , limitsDimension);
	ComputingLimitation cl;
	cl.setComputingLimitation(T*fevGlobal*numGen*numInd);
	decc_g.startTrain(fevGlobal , T , cl);
	delete[] limitsDimension;
	changeW(decc_g.getGlobalSolution());
	
	ADAMTrain(x , y , dataSize , numEpochs , beta1 , beta2 , learningRate);
	cout << "Error after training: " << getError(x , y , dataSize) <<  endl;
}


double NeuronNetwork::getError(double** x, double* y, int dataSize)
{
	double sum = 0;//RMSE
	for (int i = 0; i < dataSize; i++) {
		sum += pow(getValue(x[i]) - y[i], 2);
	}
	sum = pow(sum / dataSize , 0.5);
	return sum;
}

NeuronNetwork::NeuronNetwork(int neuronCount, int layerCount, int inCount, int numUsedFuncActivation):layerCount(layerCount), neuronCount(neuronCount), inCount(inCount)
{
	

	//Создание выходных коэффициентов
	outCoef = new double[neuronCount + 1];
	Neuron neuronOut(funcActivation[numUsedFuncActivation], neuronCount);//Сделано чтобы получить рандомные коэф для выходного вектора
	for (int i = 0; i < neuronCount; i++) {
		outCoef[i] = neuronOut.getW()[i];
	}

		
	//Создание нейронов в скрытых слоях
	grid = new Neuron * [layerCount];
	for (int i = 0; i < layerCount; i++) {
		grid[i] = new Neuron[neuronCount];
	}

	//Создание первого слоя
	for (int i = 0; i < neuronCount; i++) {
		Neuron neuron(funcActivation[numUsedFuncActivation], inCount);
		grid[0][i] = neuron;
	}

	//Создание скрытых слоев
	for (int i = 1; i < layerCount ; i++) {
		for (int j = 0; j < neuronCount; j++) {
			Neuron neuron(funcActivation[numUsedFuncActivation], neuronCount);
			grid[i][j] = neuron;
		}

	}

}





void NeuronNetwork::ADAMTrain(double** x , double* y , int dataSize , int numEpochs , double beta1 , double beta2 , double learningRate)
{
	function<double(double*)> funcOptimization = [&](double* w) { changeW(w);	return getError(x , y , dataSize); };
	int amountW = (neuronCount * (inCount + 1) + (layerCount - 1) * (neuronCount + 1) * neuronCount + (neuronCount + 1));
	double* w = new double[amountW];
	//получение начальных весов из нейронов
	int counter = 0;
	for (int i = 0; i < layerCount; i++) {
		for (int j = 0; j < neuronCount; j++) {
			double* wNeuron = grid[i][j].getW();
			for (int k = 0; k < ((i == 0) ? (inCount + 1) : (neuronCount + 1)); k++) {
				w[counter] = wNeuron[k];
				counter++;
			}
		}
	}
	//выходные коэффициенты
	for (int i = 0; i < neuronCount + 1; i++) {
		w[counter] = outCoef[i];
		counter++;
	}
	
	//ADAM optimization
	double* v_t = new double[amountW];
	double* s_t = new double[amountW];
	for (int i = 0; i < amountW; i++) {
		v_t[i] = 0;
		s_t[i] = 0;
	}
	for (int epoch = 0; epoch < numEpochs; epoch++) {
		
		for (int derivativeIndex = 0; derivativeIndex < amountW; derivativeIndex++) {
			double h = 0.00001;
			double* wPlus = new double[amountW];
			double* wMinus = new double[amountW];
			for(int i = 0 ; i < amountW ; i++) {
				wPlus[i] = w[i];
				wMinus[i] = w[i];
			}
			wPlus[derivativeIndex] += h;
			wMinus[derivativeIndex] -= h;
			double derivative = (funcOptimization(wPlus) - funcOptimization(wMinus)) / (2 * h);
			v_t[derivativeIndex] = beta1 * v_t[derivativeIndex] - (1 - beta1) * derivative;
			s_t[derivativeIndex] = beta2 * s_t[derivativeIndex] + (1 - beta2) * derivative * derivative;
			

			w[derivativeIndex] += learningRate * (v_t[derivativeIndex] / (1 - pow(beta1 , epoch + 1))) / (sqrt(s_t[derivativeIndex] / (1 - pow(beta2 , epoch + 1))) + 1);
			if (isnan(w[derivativeIndex])) {
				cout<<s_t[derivativeIndex]<<endl;
				cout << sqrt(s_t[derivativeIndex] / (1 - pow(beta2 , epoch + 1))) + 1 << endl;
				cout << w[derivativeIndex] << endl;
				throw "Error";

			}
			delete[] wPlus;
			delete[] wMinus;
		}
	}
	delete[]v_t;
	delete[]s_t;
	delete[]w;
}

double NeuronNetwork:: getValue(double* in)
{
	//double* tmpResultsHL = new double[neuronCount]; //Временный массив для хранения результатов 
	//Первый слой
	double* layerValues = matrixMulti(grid[0], in, inCount);
	//tmpResultsHL = layerValues;

	//Цикл прогонки по нейронам
	for (int i = 1; i < layerCount; i++) {

		//tmpResultsHL = matrixMulti(grid[i] , tmpResultsHL, neuronCount);
		double* tmp = layerValues;
		layerValues = matrixMulti(grid[i], layerValues, neuronCount);
		delete[] tmp;
	}

	//Цикл получения выходящего значения
	double sum = 0;//Здесь будет хранится значение у
	for (int i = 0; i < neuronCount; i++) {
		//sum += outCoef[i] * tmpResultsHL[i];
		sum += outCoef[i] * layerValues[i];
	}
	sum += outCoef[neuronCount];//Добавление смещения
	//delete[] tmpResultsHL;
	delete[] layerValues;
	return sum;
}



