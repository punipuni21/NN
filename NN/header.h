#pragma once
#include <stdio.h>
#include <time.h>
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
#define Threshold 0.01	//�������l
#define COUNTMAX 5000	//�ő厎�s��

extern double x[IN_NUM];
extern double y[OUT_NUM];
extern double w[LAYER_NUM + 1][NMAX][NMAX];
extern int D[LAYER_NUM + 2];	//�e�w�̃f�[�^��
extern double a[LAYER_NUM + 2][NMAX];	//�e�w�ւ̓���
extern double z[LAYER_NUM + 2][NMAX];	//�e�w����̏o��
extern double tIn[DATASET_NUM][NMAX];	//���t�f�[�^(����)
extern double tOut[DATASET_NUM][NMAX];	//���t�f�[�^(�o��)

using namespace std;

double sigmoid(double x);	//�������֐�(�V�O���C�h�֐�)
void serial(double studyRate);	//�����w�K
void lump(double studyRate);	//�ꊇ�w�K
double forward(int n);	//���`�d
void backwordS(int n, double studyRate);	//�t�덷�`�d(����)
void wInit();	//�d�ݏ������֐�