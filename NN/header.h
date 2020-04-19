#pragma once
#include <stdio.h>
#include <time.h>
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
#define Threshold 0.01	//しきい値
#define COUNTMAX 5000	//最大試行回数

extern double w[LAYER_NUM + 1][NMAX][NMAX];//w[l][i][j]:l層i番目の出力を(l+1)層j番目の入力に伝達するときの重み
extern int D[LAYER_NUM + 2];	//各層のデータ数
extern double a[LAYER_NUM + 2][NMAX];	//a[l][i]:l層i番目の素子への入力(前層からの出力*重みの和)
extern double z[LAYER_NUM + 2][NMAX];	//z[l][i]:l層i番目の素子からの出力(aを活性化関数に通したもの)
extern double tIn[DATASET_NUM][NMAX];	//教師データ(入力)
extern double tOut[DATASET_NUM][NMAX];	//教師データ(出力)

using namespace std;

double sigmoid(double x);	//活性化関数(シグモイド関数)
void serial(double studyRate);	//逐次学習
void lump(double studyRate);	//一括学習
double forward(int n);	//順伝播
void backwordS(int n, double studyRate);	//逆誤差伝播(逐次)
void wInit();	//重み初期化関数