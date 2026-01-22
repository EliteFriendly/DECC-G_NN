#pragma once
#include "../general/general_var.h"
#include "../general/sample_storage.h"
#include "../general/computing_limitation.h"
#include <functional>
#include <vector>
#include <algorithm>
#include "../Diff_evolution/IndividualDiffEvolution.h"
#include "../Diff_evolution/DiffEvolution.h"

using namespace std;
class Decc_g
{

private:
    string aim = "min";
    
    int** matrixSubcomponents = nullptr;//Store indexes of dimensions for every subcomponent
    int numSubcomponents , numComponents;//Number of dimensions in subcomponent and number of subcomponents
    function<double(double*)> func;//Function to be optimized (maximization)
    double* limitsDimension = nullptr;//Only for DE
    int numDimenisons;

    IndividualDiffEvolution* bestIndividualComponent = nullptr;//Best individuals of DE for every component

    double *globalSolution = nullptr; //Global solution of DECC-G
    double globalFitness = -999999; //Global fitness of DECC-G

    double* createLocalSolution(int numComponent , double* x);

    void recreateWorstComponents();
public:

    Decc_g(int numComponents, int numDimenisons, function<double(double*)>func,double *limitsDimension):numComponents(numComponents),numDimenisons(numDimenisons),func(func)
    {
        this->limitsDimension = new double[numDimenisons*2];
        for (int i = 0; i < numDimenisons*2; i++)
        {
            this->limitsDimension[i] = limitsDimension[i];
        }
        if (numDimenisons % numComponents != 0)
        {
            throw logic_error("DECC_g error: Number of dimensions is not multiple to number of subcomponents");
            exit(1);
        }
        numSubcomponents = numDimenisons / numComponents;
    }
    


    void startTrain(int fevGlobal , int T , ComputingLimitation& cl, string aim = "min");

    double *getGlobalSolution()
    {
        return globalSolution;
    }


    ~Decc_g()
    {
        if (limitsDimension != nullptr)
        {
            delete[] limitsDimension;
            limitsDimension = nullptr;
        }
        if (matrixSubcomponents != nullptr)
        {
            for (int i = 0; i < numComponents; i++)
            {
                delete[] matrixSubcomponents[i];
            }
            delete[] matrixSubcomponents;
            matrixSubcomponents = nullptr;
        }
        if (bestIndividualComponent != nullptr)
        {
            delete[] bestIndividualComponent;
            bestIndividualComponent = nullptr;
        }
        if (globalSolution != nullptr)
        {
            delete[] globalSolution;
            globalSolution = nullptr;
        }


    }
};