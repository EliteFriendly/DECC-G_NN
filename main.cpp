#include <iostream>
#include "DECC_G/Decc_g.h"
#include "Neural_network/NeuronNetwork.h"
#include "general/general_var.h"
#include <fstream>
#include <thread>

int dimension = 1;

double researchFunction(double* x)
{
    //Example: Sphere function
    double y = 2.5*sin(1.1*cos(1.1*x[0]+2)*x[0]+5) + 7.3;
    return y;
}

void doResearch(int fevGlobal, int T, int layerCount, int neuronCount, int runNumber)
{
    double leftBound = -2 , rightBound = 5;
    double h = 0.05;
    int dataSize = (rightBound - leftBound) / h;
    double** x = new double* [dataSize];
    double* y = new double[dataSize];

    double** testX = new double* [int(dataSize*0.25)];
    double** trainX = new double* [int(dataSize * 0.75)];
    double* testY = new double[int(dataSize * 0.25)];
    double* trainY = new double[int(dataSize * 0.75)];



    int i1 = 0 , i2 = 0;
    
    for (int i = 0; i < dataSize; i++)
    {
        x[i] = new double[dimension];
        x[i][0] = leftBound + i * h;
        
        double res = researchFunction(x[i]);
        double g = (rand() % 21-10)/100.0;
        y[i] = res + res * g; //adding noise
        if (i%4 == 0)
        {
            testX[i1] = new double[dimension];
            testX[i1] = x[i];
            testY[i1] = y[i];
            i1++;
        }
        else
        {
            trainX[i2] = new double[dimension];
            trainX[i2] = x[i];
            trainY[i2] = y[i];
            i2++;
        }
    }

    ofstream file_test("results/results_test_fevG" + to_string(fevGlobal) + "_T" + to_string(T) + "_lCount" + to_string(layerCount) + "_nCount" + to_string(neuronCount) + "_run" + to_string(runNumber) + ".txt");

    ofstream file_train("results/results_train_fevG" + to_string(fevGlobal) + "_T" + to_string(T) + "_lCount" + to_string(layerCount) + "_nCount" + to_string(neuronCount) + "_run" + to_string(runNumber) + ".txt");


    if (!file_test.is_open())
    {
        throw "Error with creating file";
    }
    NeuronNetwork nn(layerCount , neuronCount , dimension , 12);
    nn.startTrain(trainX , trainY , dataSize * 0.75 , 500 , 0.9 , 0.999 , 0.15 , fevGlobal , T);
    for (int i = 0; i < dataSize * 0.25; i++)
    {
        double pred = nn.getValue(testX[i]);
        file_test << testX[i][0] << "\t" << testY[i] << "\t" << pred << "\n";
    }
    file_test.close();

    for (int i = 0; i < dataSize * 0.75; i++)
    {
        double pred = nn.getValue(trainX[i]);
        file_train << trainX[i][0] << "\t" << trainY[i] << "\t" << pred << "\n";
    }
    file_train.close();

    //clear memory
    for (int i = 0; i < dataSize; i++)
    {
        delete[] x[i];
    }

    delete[] x;
    delete[] y;


}



int main()
{
    setlocale(0 , "");
    int fevGlobal[3] = { 10, 8, 5 };
    int T[3] = { 4, 5, 8 };
    int layerCount[3] = { 2, 3, 4 };
    int neuronCount[3] = { 4, 3, 2 };
    int ft , ln = 0;
    
    for (ft = 0; ft < 1; ft++) {
        for (ln = 0; ln < 3; ln++) {
            cout << "Starting research for fevGlobal = " << fevGlobal[ft] << ", T = " << T[ft] << ", layerCount = " << layerCount[ln] << ", neuronCount = " << neuronCount[ln] << endl;
            vector<thread> threads;
            for (int runNumber = 1; runNumber <= 7; runNumber++)
                threads.push_back(thread(doResearch, fevGlobal[ft], T[ft], layerCount[ln], neuronCount[ln], runNumber));
            if (threads.size() >= 7) {
                for (auto& th : threads) {
                    th.join();
                }
            }
        }
            
    }

}
