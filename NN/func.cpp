#include "header.h"


//活性化関数(シグモイド関数)
double sigmoid(double x) {
	return 1 / (1 + exp(-x));
}

//逐次学習
void serial(double studyRate) {
	int n = 0, count = 0;
	double difference;
	//まずすべてのデータを一度通す(不要なら省く)
	for (n = 0; n < DATASET_NUM; n++) {
		forward(n);
		//逆伝播
		backwordS(n, studyRate);
	}
	while (count < COUNTMAX) {
		n = count % DATASET_NUM;	//0 ≦ n ≦ DATASET_NUM - 1
		//順伝播
		difference = forward(n);
		//誤差がしきい値未満なら処理を終了する
		if (difference < Threshold)
			break;
		//逆伝播
		backwordS(n, studyRate);
		count++;
	}
	

	return;
}

//一括学習
void lump(double studyRate) {
	int i, j, k, l, n, count = 0;
	double dNum = DATASET_NUM;
	double dLdaSum;
	double d = D[LAYER_NUM + 1];	//平均をとるため出力層の要素数使用
	double dLda[LAYER_NUM + 2][NMAX] = {0};
	double difference;
	//z[0] = x(入力値)
	while (count < COUNTMAX) {
		count++;
		//順伝播
		difference = 0;
		for (n = 0; n < DATASET_NUM; n++) {
			difference += forward(n);	//n番目のデータセットを通す
			//データセットを通すたびにdL/daを計算し、それぞれのdL/daの和をとる
			for (i = LAYER_NUM; i >= 0; i--) {
				//出力層の場合
				if (i == LAYER_NUM) {					//i:層の総数 - 2  (中間層の数)
					for (k = 0; k < D[i + 1]; k++) {	//0 ≦ k ≦(出力層のデータ数)
						for (j = 0; j < D[i]; j++) {	//0 ≦ j ≦(出力層の一つ前の層のデータ数)
							dLda[i + 1][k] += 1 / d * 2 * (z[i + 1][k] - tOut[n][k]) * z[i + 1][k] * (1 - z[i + 1][k]);	//dL/daの和をとる
						}
					}
				}
				else {	//最下層でない場合
					for (k = 0; k < D[i + 1]; k++) {
						for (j = 0; j < D[i]; j++) {
							dLdaSum = 0;
							for (l = 0; l < D[i + 2]; l++)
								dLdaSum += dLda[i + 2][l] * w[i + 1][k][l];
							dLda[i + 1][k] += z[i + 1][k] * (1 - z[i + 1][k]) * dLdaSum;
						}
					}
				}
			}
		}

		//誤差がしきい値未満なら処理を終了する
		if (difference < Threshold)
			break;

		//すべて足し合わせたのでdLdaは平均して扱う(dNumで割る)
		for (i = LAYER_NUM; i >= 0; i--) {
			//最下層の場合
			if (i == LAYER_NUM) {
				for (k = 0; k < D[i+1]; k++) {
					for (j = 0; j < D[i]; j++) {
						w[i][j][k] -= studyRate * z[i][j] * dLda[i + 1][k] / dNum;
					}
				}
			}
			else {	//最下層でない場合
				for (k = 0; k < D[i + 1]; k++) {
					for (j = 0; j < D[i]; j++) {
						w[i][j][k] -= studyRate * z[i][j] * dLda[i + 1][k] / dNum;
					}
				}
			}
		}
	}
	return;
}

//順伝播
double forward(int n) {	//n:データセットの番号
	int i, j, k;
	double difference = 0;
	for (i = 0; i < IN_NUM; i++)
		z[0][i] = tIn[n][i];	//教師データをz[0]に代入

	for (i = 1; i < (LAYER_NUM + 2); i++) {
		for (j = 0; j < D[i]; j++) {
			a[i][j] = 0;
			for (k = 0; k < D[i - 1]; k++) {
				a[i][j] += w[i - 1][k][j] * z[i - 1][k];	//各層への入力aは(重みw*前層の出力z)の和
			}
			z[i][j] = sigmoid(a[i][j]);	//出力は入力aを活性化関数に通したもの
		}
	}
	for (i = 0; i < D[LAYER_NUM + 1]; i++) {
		difference += pow(z[LAYER_NUM + 1][i] - tOut[n][i], 2);	//教師データとの差分の二乗を返す
	}
	return difference;
}

//逆誤差伝播(逐次)
void backwordS(int n, double studyRate) {
	int i, j, k, l;
	double dLdaSum;
	double d = D[LAYER_NUM + 1];	//平均をとるため出力層の要素数使用
	double dLda[LAYER_NUM+2][NMAX];

	for (i = LAYER_NUM; i >= 0; i--) {
		//最下層の場合
		if (i == LAYER_NUM) {
			for (k = 0; k < D[i+1]; k++) {
				for (j = 0; j < D[i]; j++) {
					dLda[i + 1][k] = 1 / d * 2 * (z[i + 1][k] - tOut[n][k]) * z[i + 1][k] * (1 - z[i + 1][k]);
					w[i][j][k] -=  studyRate * z[i][j] * dLda[i+1][k];
				}
			}
		}
		else {	//最下層でない場合
			for (k = 0; k < D[i + 1]; k++) {
				for (j = 0; j < D[i]; j++) {
					dLdaSum = 0;
					for (l = 0; l < D[i + 2]; l++)
						dLdaSum += dLda[i + 2][l] * w[i + 1][k][l];
					dLda[i + 1][k] = z[i+1][k]*(1-z[i+1][k])*dLdaSum;
					w[i][j][k] -= studyRate * z[i][j] * dLda[i + 1][k];
				}
			}
		}
	}

}

//重み初期化関数
void wInit() {	//重み初期化関数
	srand(time(NULL));

	for (int i = 0; i < (LAYER_NUM + 1); ++i) {
		for (int j = 0; j < NMAX; j++) {
			for (int k = 0; k < NMAX; k++) {
				w[i][j][k] = (double)rand() / RAND_MAX;
			}
		}
	}
}