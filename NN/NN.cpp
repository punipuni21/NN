#pragma warning(disable: 4996)

#include "header.h"

double x[IN_NUM];
double y[OUT_NUM];
double w[LAYER_NUM + 1][NMAX][NMAX];
int D[LAYER_NUM + 2] = { 3,4,1 };	//各層のデータ数
double a[LAYER_NUM + 2][NMAX];	//各層への入力
double z[LAYER_NUM + 2][NMAX];	//各層からの出力
double tIn[DATASET_NUM][NMAX];	//教師データ(入力)
double tOut[DATASET_NUM][NMAX];	//教師データ(出力)

double abc = 0;

int main(void)
{

	FILE* fp1;
	FILE* fp2;

	int Flag = TRUE;
	double studyRate = 0.1;

	//alternative
	D[0] = 3, D[1] = 4, D[2] = 1;
	
	if ((fp1 = fopen("test_in.csv", "r")) == NULL) {
		cout << "Training input File Open ERROR!" << endl;
		return 0;
	}
	if ((fp2 = fopen("test_out.csv", "r")) == NULL) {
		cout << "Training output File Open ERROR!" << endl;
		return 0;
	}
	for (int i = 0; i < DATASET_NUM; i++) {
		for (int j = 0; j < IN_NUM; j++) {
			fscanf(fp1, "%lf", &tIn[i][j]);
		}
	}
	for (int i = 0; i < DATASET_NUM; i++) {
		for (int j = 0; j < OUT_NUM; j++) {
			fscanf(fp2, "%lf", &tOut[i][j]);
		}
	}

	//重みの初期化
	wInit();
	double difference = 0;
	cout << "初期状態:--------------------------------------------------" << endl;
	for (int i = 0; i < DATASET_NUM; i++) {
		difference += forward(i);
		cout <<"教師出力 (" << i+1 << "):";
		for (int j = 0; j < OUT_NUM; j++)
			cout << tOut[i][j] << " ";
		cout << endl;
		cout << "計算結果 (" << i+1 << "):";
		for (int j = 0; j < OUT_NUM; j++)
			cout << z[LAYER_NUM + 1][j] << " ";
		cout << endl;
	}
	cout << "差分:" << difference << endl;
	cout << "------------------------------------------------------------" << endl;
	lump(studyRate);
	difference = 0;
	cout << "ＮＮ結果:--------------------------------------------------" << endl;
	for (int i = 0; i < DATASET_NUM; i++) {
		difference += forward(i);
		cout << "教師出力 (" << i + 1 << "):";
		for (int j = 0; j < OUT_NUM; j++)
			cout << tOut[i][j] << " ";
		cout << endl;
		cout << "計算結果 (" << i + 1 << "):";
		for (int j = 0; j < OUT_NUM; j++)
			cout << z[LAYER_NUM + 1][j] << " ";
		cout << endl;
	}
	cout << "差分:" << difference << endl;
	cout << "------------------------------------------------------------" << endl;
    return 0;
}