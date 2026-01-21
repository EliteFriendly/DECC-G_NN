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

    
    int** matrixSubcomponents = nullptr;//Store indexes of dimensions for every subcomponent
    int numSubcomponents , numComponents;//Number of dimensions in subcomponent and number of subcomponents
    function<double(double*)> func;//Function to be optimized (maximization)
    double* limitsDimension = nullptr;//Only for DE
    int numDimenisons;

    IndividualDiffEvolution* bestIndividualComponent = nullptr;//Best individuals of DE for every component

    double *globalSolution = nullptr; //Global solution of DECC-G


    double* createLocalSolution(int numComponent , double* x);

    void recreateWorstComponents()
    {
        //create array of fitness and indexes
        vector<pair<double, int>> fitnessComponents;
        for (int i = 0; i < numComponents; i++)
        {
            fitnessComponents.push_back({ bestIndividualComponent[i].getFitness() , i });
        }
        //sort by fitness
        sort(fitnessComponents.begin() , fitnessComponents.end() , [](const pair<double , int>& a , const pair<double , int>& b) { return a.first < b.first; });

        //Choose worst components to recreate
        int amToRecreate = (numComponents / 2);
        int* arrOldSubComponents = new int[amToRecreate * numSubcomponents];

        //store old subcomponents
        for (int i = 0; i < amToRecreate; i++)
        {
            for (int j = 0; j < numSubcomponents; j++)
            {
                arrOldSubComponents[i * numSubcomponents + j] = matrixSubcomponents[fitnessComponents[i].second][j];
            }
        }
        //create random numbers for new subcomponents
        vector<int> arrIndexDimension(amToRecreate * numSubcomponents);
        iota(arrIndexDimension.begin(), arrIndexDimension.end(), 0);
        shuffle(arrIndexDimension.begin() , arrIndexDimension.end() , gen);

        //recreate worst components
        for (int i = 0; i < amToRecreate; i++)
        {
            for (int j = 0; j < numSubcomponents; j++)
            {
                matrixSubcomponents[fitnessComponents[i].second][j] = arrOldSubComponents[arrIndexDimension[i * numSubcomponents + j]];
            }
        }
    }
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
            throw logic_error("Error: Number of dimensions is not multiple to number of subcomponents");
            exit(1);
        }
        numSubcomponents = numDimenisons / numComponents;
    }
    


    void startTrain(int fevGlobal , int T , ComputingLimitation& cl);

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