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

#define IN_NUM 3    //入力数
#define OUT_NUM 1   //出力数
#define MIDDLE_NUM 4	//中間層ノード数
#define LAYER_NUM 1 //中間層数
#define M_NUM 4 //中間層の素子数
#define NMAX 10  //ノードの最大数
#define DATASET_NUM 6 //データセット数
#define STUDYRATE 0.1	//学習率
#define Threshold 0.01	//しきい値
#define COUNTMAX 5000	//最大試行回数
#define DIS_DATASET_NUM 2 //データセット数

using namespace std;

double sigmoid(double x);	//活性化関数(シグモイド関数)
void serial(vector<vector<vector<double>>>& w, vector<vector<double>> tIn, vector<vector<double>> tOut);	//逐次学習
void lump(vector<vector<vector<double>>>& w, vector<vector<double>> tIn, vector<vector<double>> tOut);	//一括学習
double forward(int n, vector<vector<double>> tIn, vector<vector<double>> tOut, vector<vector<vector<double>>>& a, vector<vector<vector<double>>>& z, vector<vector<vector<double>>> w);	//順伝播
void wInit(vector<vector<vector<double>>>& w);	//重み初期化関数