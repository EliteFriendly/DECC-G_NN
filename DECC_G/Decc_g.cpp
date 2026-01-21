#include "Decc_g.h"

double* Decc_g::createLocalSolution(int numComponent , double* x)
{
    double* localSolution = new double[numDimenisons];
    for (int i = 0; i < numDimenisons; i++)
    {
        localSolution[i] = globalSolution[i];
    }
    for (int i = 0; i < numSubcomponents; i++)
    {
        localSolution[matrixSubcomponents[numComponent][i]] = x[i];
    }
    return localSolution;
}

void Decc_g::startTrain(int fevGlobal , int T , ComputingLimitation& cl)
{
    //create random numbers for subcomponents
    vector<int> arrIndexDimension(numDimenisons);
    iota(arrIndexDimension.begin(), arrIndexDimension.end(), 0);
    shuffle(arrIndexDimension.begin() , arrIndexDimension.end() , gen);

    //create global component
    globalSolution = new double[numDimenisons];
    for (int i = 0; i < numDimenisons; i++)
    {
        globalSolution[i] = limitsDimension[2 * i] + (limitsDimension[2 * i + 1] - limitsDimension[2 * i]) * (gen() % 1000) / 1000.0;
    }


    //create subcomponents
    matrixSubcomponents = new int* [numComponents];
    for (int i = 0; i < numComponents; i++)
    {
        matrixSubcomponents[i] = new int[numSubcomponents];
        for (int j = 0; j < numSubcomponents; j++)
        {
            matrixSubcomponents[i][j] = arrIndexDimension[i * numSubcomponents + j];
        }
    }

    bestIndividualComponent = new IndividualDiffEvolution[numComponents];

    //train subcomponents
    for (int fev = 0; fev < fevGlobal; fev++)
    {
        for (int fevLocal = 0; fevLocal < T; fevLocal++)
        {

            for (int i = 0; i < numComponents; i++)
            {
                //create local limits
                double* localLimitsDimension = new double[numSubcomponents * 2];
                for (int j = 0; j < numSubcomponents; j++)
                {
                    localLimitsDimension[2 * j] = limitsDimension[2 * matrixSubcomponents[i][j]];
                    localLimitsDimension[2 * j + 1] = limitsDimension[2 * matrixSubcomponents[i][j] + 1];
                }
                //create fitness function
                function<double(double*)> fitnessFunc = [&](double* x) {
                    double* localSolution = createLocalSolution(i , x);
                    double result = func(localSolution);
                    delete[] localSolution;
                    return result;
                    };

                //Create and train DE for component
                DiffEvolution DE(fitnessFunc , localLimitsDimension , numSubcomponents , "best1" , "max");
                DE.startSearch(1e-6 , 0.5 , 0.5 , numInd , numGen , cl);
                if (bestIndividualComponent[i].getFitness() < DE.getBestIndividual().getFitness())
                {
                    bestIndividualComponent[i] = DE.getBestIndividual();
                }

            }
        }
        //recreate worst components/2
        recreateWorstComponents();
        //update final solution
        for (int i = 0; i < numComponents; i++)
        {
            double* compCoord = bestIndividualComponent[i].getCoordinats();
            for (int j = 0; j < numSubcomponents; j++)
            {
                globalSolution[matrixSubcomponents[i][j]] = compCoord[j];
            }
        }
        

    }

}