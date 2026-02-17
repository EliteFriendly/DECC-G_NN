#include <iostream>
#include "DECC_G/Decc_g.h"
#include "Neural_network/NeuronNetwork.h"
#include "general/general_var.h"
#include <fstream>
#include <string>
#define _CRTDBG_MAP_ALLOC
#include <crtdbg.h>
#include <ctime>
#include <thread>




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

void doResearch(string path, int dim, int fevGlobal, int T, int layerCount, int neuronCount,string task, string run, int amWeight, int amComponents)
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


    double** trainX = new double*[storage.getTrainSize()];
    double** testX = new double* [storage.getTestSize()];
    double* trainY = new double[storage.getTrainSize()];
    double* testY = new double[storage.getTestSize()];
    for (int i = 0; i < storage.getTrainSize(); i++)
    {
        trainX[i] = new double[dim];
        for (int j = 0; j < dim; j++)
        {
            trainX[i][j] = storage.getTrainData()[i][j];
        }
        trainY[i] = storage.getTrainData()[i][dim];
    }
    for (int i = 0; i < storage.getTestSize(); i++)
    {
        testX[i] = new double[dim];
        for (int j = 0; j < dim; j++)
        {
            testX[i][j] = storage.getTestData()[i][j];
        }
        testY[i] = storage.getTestData()[i][dim];
    }

    

    ofstream file_test("results/" + task + "_test_fevG" + to_string(fevGlobal) + "_T" + to_string(T) + "_lCount" + to_string(layerCount) + "_nCount" + to_string(neuronCount) + "_run" + run + ".txt");

    ofstream file_train("results/" + task + "_train_fevG" + to_string(fevGlobal) + "_T" + to_string(T) + "_lCount" + to_string(layerCount) + "_nCount" + to_string(neuronCount) + "_run" + run + ".txt");


    if (!file_test.is_open())
    {
        throw "Error with creating file";
    }
    NeuronNetwork nn(layerCount , neuronCount , dim, 12);
    int ADAMcalc = 10800;
    if (numGen == 0) {
        ADAMcalc += round((ADAMcalc*amComponents) / double(amWeight));
    }
    nn.startTrain(trainX , trainY , storage.getTrainSize() , ADAMcalc , 0.9 , 0.999 , 0.15 , fevGlobal , T);

    for (int i = 0; i < dim+2; i++)
    {
            if(i == dim){
                file_test << "Y_true" << "\t";
                file_train << "Y_true" << "\t";
            }
        else if (i == dim + 1){
            file_test << "Y_pred" << "\t";
            file_train << "Y_pred" << "\t";
        }
        else {
                file_test << "X" << i << "\t";
                file_train << "X" << i << "\t";
            }
    }
    file_test << "\n";
    file_train << "\n";

    for (int i = 0; i < storage.getTestSize(); i++)
    {
        
        double pred_test = nn.getValue(testX[i]);
        double pred_train = nn.getValue(trainX[i]);
        for (int j = 0;j < dim;j++) {
            file_test << testX[i][j] << "\t";
            file_train << trainX[i][j] << "\t";
        }
        file_test << testY[i] << "\t" << pred_test << "\n";
        file_train << trainY[i] << "\t" << pred_train << "\n";
    }
    file_test.close();
    file_train.close();
    //clering memory
    for (int i = 0;i < dim;i++) {
        delete[] trainX[i];
        delete[] testX[i];
    }
    delete[] trainX;
    delete[] testX;
    delete[] trainY;
    delete[] testY;


}





int main()
{
    // Включаем автоматический дамп утечек при выходе
    //_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);


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

    
    clock_t start = clock();
    setlocale(0 , "");
    int fevGlobal[3] = { 6, 8, 5 };
    int T[3] = { 2, 5, 8 };
    int layerCount[3] = { 2, 3, 4 };// for 2, 3 and 4 parameters
    int neuronCount[3] = { 4, 3, 2 };// for 2, 3 and 4 parameters
    int amWeights[3] = { 27,40,45 };
    int ammComponents[3] = { 3,10,9 };
    // amm parametrs for 2, 3 and 4 parameters are next: 27, 40, 45
    int ln = 0;
    //cout<< "test_task/"+file_names[0]<<endl;
    try {
        for (int i = 0;i < parameter_counts.size();i++) {
            vector<thread> threads;
            for (int runNumber = 1; runNumber <= 8; runNumber++) {
                cout << "Running task: " << file_names[i] << ", run: " << runNumber << endl;
                switch (parameter_counts[i]) {
                    case 2:
                        ln = 0;
                        break;
                    case 3:
                        ln = 1;
                        break;
                    case 4:
                        ln = 2;
                        break;
                    default:
                        throw "Unsupported number of parameters";
                }
                
                if (threads.size() <= 8) {
                    threads.push_back(thread(doResearch , "test_task/" + file_names[i] , parameter_counts[i] ,
                    fevGlobal[0] , T[0] , layerCount[ln] , neuronCount[ln] ,
                        file_names[i].substr(0 , file_names[i].find('.')) , to_string(runNumber),amWeights[ln],ammComponents[ln]));
                    
                }
                else {
                    for (int i = 0; i < 8; i++) {
                        threads[i].join();
                    }
                    threads.clear();
                }

                /*doResearch("test_task/" + file_names[i] , parameter_counts[i] , fevGlobal[ft] ,
                    T[ft] , layerCount[ln] , neuronCount[ln] ,
                        file_names[i].substr(0 , file_names[i].find('.')) , to_string(runNumber));*/
            }
            if (threads.size() > 0) {
                for (int i = 0; i < threads.size(); i++) {
                    threads[i].join();
                }
            }
        }

                
                //vector<thread> threads;
                
                //threads.push_back(thread(doResearch , fevGlobal[ft] , T[ft] , layerCount[ln] , neuronCount[ln] , runNumber));
            /*if (threads.size() >= 7) {
                for (auto& th : threads) {
                    th.join();
                }
            }*/

            

        
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
