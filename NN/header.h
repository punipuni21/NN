#pragma once
#include <stdio.h>
#include <time.h>
#include <iostream>
#include <cmath>
#include <vector>
#include <random>
#include <fstream>
#include <string>
#include <sstream>

#define TRUE 0
#define FALSE 1

#define IN_NUM 3    //���͐�
#define OUT_NUM 1   //�o�͐�
#define MIDDLE_NUM 4	//���ԑw�m�[�h��
#define LAYER_NUM 1 //���ԑw��
#define M_NUM 4 //���ԑw�̑f�q��
#define NMAX 10  //�m�[�h�̍ő吔
#define DATASET_NUM 6 //�f�[�^�Z�b�g��
#define STUDYRATE 0.1	//�w�K��
#define Threshold 0.01	//�������l
#define COUNTMAX 5000	//�ő厎�s��
#define DIS_DATASET_NUM 2 //�f�[�^�Z�b�g��

using namespace std;

double sigmoid(double x);	//�������֐�(�V�O���C�h�֐�)
void serial(vector<vector<vector<double>>>& w, vector<vector<double>> tIn, vector<vector<double>> tOut);	//�����w�K
void lump(vector<vector<vector<double>>>& w, vector<vector<double>> tIn, vector<vector<double>> tOut);	//�ꊇ�w�K
double forward(int n, vector<vector<double>> tIn, vector<vector<double>> tOut, vector<vector<vector<double>>>& a, vector<vector<vector<double>>>& z, vector<vector<vector<double>>> w);	//���`�d
void wInit(vector<vector<vector<double>>>& w);	//�d�ݏ������֐�