#pragma once
#include <stdio.h>
#include <iostream>
#include <cmath>
#include <vector>
#include <random>

#define TRUE 0
#define FALSE 1

#define IN_NUM 3    //���͐�
#define OUT_NUM 1   //�o�͐�
#define LAYER_NUM 1 //�w��
#define M_NUM 4 //���ԑw�̑f�q��
#define NMAX 10  //�ő吔
#define DATASET_NUM 6 //�f�[�^�Z�b�g��
#define Threshold 1.0	//�������l
#define COUNTMAX 1000	//�ő厎�s��

using namespace std;

double x[IN_NUM];
double y[OUT_NUM];
double w[LAYER_NUM + 1][NMAX][NMAX];
int D[LAYER_NUM + 2] = {3,4,1};	//�e�w�̃f�[�^��
double a[LAYER_NUM+2][NMAX];	//�e�w�ւ̓���
double z[LAYER_NUM + 2][NMAX];	//�e�w����̏o��
double t[6] = { 0, 0, 1, 0, 1, 1 };


double sigmoid(double x);
double forward(int n);
void backwordS(int n, double studyRate);
void wInit();