
//#include
#include "../general/general_var.h"
#include<iostream>
#include<vector>
#include <functional>


using namespace std;

class IndividualDiffEvolution
	#pragma once
{
private:
	double* coordinats = nullptr;//Координаты индивида
	double* limitsDimension = nullptr;//Ограничение на каждую ось
	double acc;//Точность поиска точки
	double fitness = -9999999;;//Значение пригодности
	function<double(double*)> func;
	string aim;
	int ammDimen;
	

public:
	IndividualDiffEvolution(const IndividualDiffEvolution&);
	IndividualDiffEvolution() {}
	IndividualDiffEvolution(double* limitsDimension, function<double(double*)> func, int ammDimen, double acc, string aim);
	void replaceCoordinats(double* repCoord) {
		if (coordinats == nullptr) {
			coordinats = new double[ammDimen];
		}

		for (int i = 0; i < ammDimen; i++) {
			coordinats[i] = repCoord[i];
		}
	}
	double getFitness() {return fitness;}
	double* getCoordinats() { return coordinats; }


	void calcFitness() {
		
		if (aim == "max") {
			fitness = func(coordinats);
		}
		if (aim == "min") {
			fitness = 1 / (func(coordinats) + 1);
		}
	}


	~IndividualDiffEvolution()
	{
		if (coordinats != nullptr) {
			delete[] coordinats;
			coordinats = nullptr;
		}
		if (limitsDimension != nullptr) {
			delete[] limitsDimension;
			limitsDimension = nullptr;
		}
	}

	
	bool operator <(const IndividualDiffEvolution&);
	bool operator >(const IndividualDiffEvolution&);
	//bool operator ==(const IndividualDiffEvolution&);
	IndividualDiffEvolution operator +(const IndividualDiffEvolution&);
	IndividualDiffEvolution operator -(const IndividualDiffEvolution&);
	IndividualDiffEvolution operator =(const IndividualDiffEvolution&);
	IndividualDiffEvolution operator *(double);
};

