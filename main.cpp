#include <iostream>
#include "DECC_G/Decc_g.h"


const int dimension = 10;

double testFunction(double* x)
{
    //Example: Sphere function
    double sum = 0.0;
    for (int i = 0; i < dimension; i++) // Assuming 10 dimensions
    {
        sum += x[i] * x[i];
    }
    return sum;
}
int main()
{
    setlocale(0, "");
    double* limitsDimension = new double[dimension * 2];
    for (int i = 0; i < dimension; i++)
    {
        limitsDimension[2 * i] = -10.0;     // Lower bound for dimension i
        limitsDimension[2 * i + 1] = 10.0;  // Upper bound for dimension i
    }
    Decc_g decc_g(5 , dimension , testFunction , limitsDimension);
    decc_g.startTrain(10 , 5 , *(new ComputingLimitation()));
    delete[] limitsDimension;
    cout << "Результат оптимизации:" << endl;
    double* solution = decc_g.getGlobalSolution();
    for (int i = 0; i < dimension; i++)
    {
        cout << "x[" << i << "] = " << solution[i] << endl;
    }

    return 0;
}
