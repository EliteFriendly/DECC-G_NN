#include <iostream>
#include "DECC_G/Decc_g.h"
#include "Neural_network/NeuronNetwork.h"
#include "general/general_var.h"
#include <fstream>
#include <string>
#define _CRTDBG_MAP_ALLOC
#include <crtdbg.h>
#include <ctime>




#ifdef _DEBUG
#define new new(_NORMAL_BLOCK, __FILE__, __LINE__)
#endif

int dimension = 1;
static const int DATA_SIZE = 1000;
double researchFunction(double* x)
{
    //Example: Sphere function
    double y = 2.5*sin(1.1*cos(1.1*x[0]+2)*x[0]+5) + 7.3;
    return y;
}

void doResearch(string path, int dim, int fevGlobal, int T, int layerCount, int neuronCount, int runNumber)
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

    nn.startTrain(trainX , trainY , int(dataSize * 0.75) , 500 , 0.9 , 0.999 , 0.15 , fevGlobal , T);
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
    
    delete[] testY;
    delete[] trainY;
    delete[] testX;
    delete[] trainX;
    delete[] x;
    delete[] y;

    ifstream file(path);
    if (!file.is_open())
    {
        cerr << "Error opening files" << endl;
        exit(1);
    }
    // Read data from file
    double **data = new double *[DATA_SIZE];
    for (int i = 0; i < DATA_SIZE; i++)
    {

        data[i] = new double[dim+1];

        for (int j = 0; j < dim+1; j++)
        {
            file >> data[i][j];
            //cout<<data[i][j]<<" ";
            if (file.peek() == ',')
                file.ignore();
            // cout << data[i][j] << " ";
        }
        //cout << endl;

        // cout << endl;
    }
    file.close();
    SampleStorage storage(DATA_SIZE, dim, data, 0.75, "reg"); // 75% for training

    int treeDepth = 4; // depth of tree

    ofstream fileOut("algorithm_results/Results/Best_" + to_string(runNumber) + ".txt");
    if (!fileOut.is_open())
    {
        cout << "Error opening file out" << endl;
        exit(1);
    }
    ofstream filePoints("algorithm_results/Points/" + to_string(runNumber) + ".txt");
    if (!filePoints.is_open())
    {
        cout << "Error opening file out" << endl;
        exit(1);
    }

    cout << "Iteration " << to_string(runNumber) << endl;
    //AdaptiveGeneticProgramming proba(treeDepth, "reg");
    //proba.numFileAndTrail(mark,true);
    //proba.startTrain(data, dim, 1, DATA_SIZE,30,30);
    //Tree best = proba.getBest();
    // fileOut << proba.getError(dataTest, size * 0.25) << endl;
    //fileOut << best.getFunc() << endl;
    //fileOut << best.getMatrix() << endl;
    //fileOut.close();
    for (int i = 0;i < storage.getTestSize();i++) {
        ///filePoints << storage.getTestData()[i][dim] << " " << best.getValue(storage.getTestData()[i])[0] << endl;
    }
    filePoints.close();


}

void test(string path, int dim, string mark)
{
    ifstream file(path);
    if (!file.is_open())
    {
        cerr << "Error opening files" << endl;
        exit(1);
    }
    // Read data from file
    double **data = new double *[DATA_SIZE];
    for (int i = 0; i < DATA_SIZE; i++)
    {

        data[i] = new double[dim+1];

        for (int j = 0; j < dim+1; j++)
        {
            file >> data[i][j];
            //cout<<data[i][j]<<" ";
            if (file.peek() == ',')
                file.ignore();
            // cout << data[i][j] << " ";
        }
        //cout << endl;

        // cout << endl;
    }
    file.close();
    SampleStorage storage(DATA_SIZE, dim, data, 0.75, "reg"); // 75% for training

    int treeDepth = 4; // depth of tree

    ofstream fileOut("algorithm_results/Results/Best_" + mark + ".txt");
    if (!fileOut.is_open())
    {
        cout << "Error opening file out" << endl;
        exit(1);
    }
    ofstream filePoints("algorithm_results/Points/" + mark + ".txt");
    if (!filePoints.is_open())
    {
        cout << "Error opening file out" << endl;
        exit(1);
    }

    cout << "Iteration " << mark << endl;
    //AdaptiveGeneticProgramming proba(treeDepth, "reg");
    //proba.numFileAndTrail(mark,true);
    //proba.startTrain(data, dim, 1, DATA_SIZE,30,30);
    //Tree best = proba.getBest();
    // fileOut << proba.getError(dataTest, size * 0.25) << endl;
    //fileOut << best.getFunc() << endl;
    //fileOut << best.getMatrix() << endl;
    //fileOut.close();
    for (int i = 0;i < storage.getTestSize();i++) {
        //filePoints << storage.getTestData()[i][dim] << " " << best.getValue(storage.getTestData()[i])[0] << endl;
    }
    filePoints.close();
}



int main()
{
    // Включаем автоматический дамп утечек при выходе
    _CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);


setlocale(0,"");
    clock_t tStart = clock();

    std::vector<std::string> file_names = {
        "I_6_2b.txt",
        "I_8_14.txt", 
        "I_12_1.txt",
        "I_12_2.txt",
        "I_12_4.txt",
        "I_14_3.txt",
        "I_14_4.txt",
        "I_15_3x.txt",
        "I_15_10.txt",
        "I_18_4.txt",
        "I_24_6.txt",
        "I_34_8.txt"
    };

// Массив количества изменяемых параметров для каждой задачи
    std::vector<int> parameter_counts = {
        3,  // I.6.2b: 
        4,  // I.8.14: x1, y1, x2, y2
        2,  // I.12.1: q1, q2, r
        4,  // I.12.2: q, E, v, B
        3,  // I.12.4: mu, r
        3,  // I.14.3: m, h
        2,  // I.14.4: k, x
        4,  // I.15.3x: x1, u, t
        3,  // I.15.10: m, v
        4,  // I.18.4: m, v, r, theta
        4,  // I.24.6: n, theta2
        4   // I.34.8: q, a
    };
    string st = "test/" + file_names[0];
    //cout << st << endl;
    ifstream file(st);
    if (!file.is_open())
    {
        cerr << "Error opening files" << endl;
        exit(1);
    }
    


    try {
    for (int i = 0;i < file_names.size();i++) {
        for (int r = 0; r < 10; r++)
        {
            test("test/"+file_names[i], parameter_counts[i],to_string(i)+to_string(r));
        }
    }
    }
    catch(exception& e)
    {
        cout << e.what() << endl;
    }


    clock_t start = clock();
    setlocale(0 , "");
    int fevGlobal[3] = { 4, 8, 5 };
    int T[3] = { 2, 5, 8 };
    int layerCount[3] = { 2, 3, 4 };
    int neuronCount[3] = { 4, 3, 2 };
    int ft , ln = 0;
    try {
        for (ft = 0; ft < 1; ft++) {
            for (ln = 0; ln < 1; ln++) {
                cout << "Starting research for fevGlobal = " << fevGlobal[ft] << ", T = " << T[ft] << ", layerCount = " << layerCount[ln] << ", neuronCount = " << neuronCount[ln] << endl;
                //vector<thread> threads;
                for (int runNumber = 0; runNumber <= 20; runNumber++)
                    //doResearch(fevGlobal[ft] , T[ft] , layerCount[ln] , neuronCount[ln] , 1);
                //threads.push_back(thread(doResearch , fevGlobal[ft] , T[ft] , layerCount[ln] , neuronCount[ln] , runNumber));
            /*if (threads.size() >= 7) {
                for (auto& th : threads) {
                    th.join();
                }
            }*/
                    cout << "jfkgj";
            }

        }
    }
    catch (const char* msg) {
        cerr << "Error: " << msg << endl;
    }
    //_CrtDumpMemoryLeaks();
    clock_t end = clock();
    double elapsed = double(end - start) / CLOCKS_PER_SEC;
    std::cout << "Elapsed time: " << elapsed << " seconds." << std::endl;
    return 0;
}
