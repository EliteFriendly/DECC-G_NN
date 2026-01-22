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

void Decc_g::recreateWorstComponents()
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

void Decc_g::startTrain(int fevGlobal , int T , ComputingLimitation& cl, string aim)
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
                    double result = (aim == "max") ? func(localSolution) : 1 / (1 + func(localSolution)); //max or min
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
        
        //update final solution


        globalFitness = (aim == "max") ? func(globalSolution) : 1 / (1 + func(globalSolution)); //max or min
        double *localSolution = new double[numDimenisons];
        for (int i = 0; i < numComponents; i++)
        {
            for (int j = 0; j < numDimenisons; j++)
            {
                localSolution[j] = globalSolution[j];
            }
            double* compCoord = bestIndividualComponent[i].getCoordinats();
            for (int j = 0; j < numSubcomponents; j++)
            {
                localSolution[matrixSubcomponents[i][j]] = compCoord[j];
            }
            double localFitness = (aim == "max") ? func(localSolution) : 1 / (1 + func(localSolution));
            if (globalFitness < localFitness)
            {
                globalFitness = localFitness;
                for (int j = 0; j < numDimenisons; j++)
                {
                    globalSolution[j] = localSolution[j];
                }
            }

        }
        recreateWorstComponents();


    }

}