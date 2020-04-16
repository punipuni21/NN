#pragma once
#include <stdio.h>
#include <iostream>
#include <cmath>
#include <vector>
#include <random>

#define TRUE 0
#define FALSE 1

#define IN_NUM 3    //入力数
#define OUT_NUM 1   //出力数
#define LAYER_NUM 1 //層数
#define M_NUM 4 //中間層の素子数
#define NMAX 10  //最大数
#define DATASET_NUM 6 //データセット数
#define Threshold 1.0	//しきい値
#define COUNTMAX 1000	//最大試行回数

using namespace std;

double x[IN_NUM];
double y[OUT_NUM];
double w[LAYER_NUM + 1][NMAX][NMAX];
int D[LAYER_NUM + 2] = {3,4,1};	//各層のデータ数
double a[LAYER_NUM+2][NMAX];	//各層への入力
double z[LAYER_NUM + 2][NMAX];	//各層からの出力
double t[6] = { 0, 0, 1, 0, 1, 1 };


double sigmoid(double x);
double forward(int n);
void backwordS(int n, double studyRate);
void wInit();