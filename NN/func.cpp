#include "header.h"

double sigmoid(double x) {
	return 1 / (1 + exp(-x));
}

//逐次学習
void serial(double studyRate) {
	int n = 0, count = 0;
	double difference;
	//重みwの初期設定
	wInit();
	
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
	int i, j, k, l, n;
	double dNum = DATASET_NUM;
	double loss;
	double dLdaSum;
	double d = D[LAYER_NUM + 1];	//平均をとるため出力層の要素数使用
	double dLda[LAYER_NUM + 2][NMAX] = {0};
	double difference, dNum = DATASET_NUM;
	//重みwの初期設定
	wInit();
	//z[0] = x(入力値)
	while (true) {
		//順伝播
		difference = 0;
		for (n = 0; n < DATASET_NUM; n++) {
			difference += forward(n);	//n番目のデータセットを通す
			//データセットを通すたびにdL/daを計算する
			for (i = LAYER_NUM; i >= 0; i--) {
				//出力層の場合
				if (i == LAYER_NUM) {					//i:層の総数 - 2  (中間層の数)
					for (k = 0; k < D[i + 1]; k++) {	//0 ≦ k ≦(出力層のデータ数)
						for (j = 0; j < D[i]; j++) {	//0 ≦ j ≦(出力層の一つ前の層のデータ数)
							dLda[i + 1][k] += 1 / d * 2 * (z[i + 1][k] - t[n]) * z[i + 1][k] * (1 - z[i + 1][k]);
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

		difference /= dNum;	//誤差平均
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
double forward(int n) {
	int i, j, k;
	double difference = 0;
	for (i = 1; i < (LAYER_NUM + 2); i++) {
		for (j = 0; j < D[i]; j++) {
			a[i][j] = 0;
			for (k = 0; k < D[i - 1]; k++) {
				a[i][j] += w[i - 1][k][j] * z[i - 1][k];
			}
			z[i][j] = sigmoid(a[i][j]);
		}
	}
	for (i = 0; i < D[LAYER_NUM + 1]; i++) {
		difference += pow(z[LAYER_NUM + 1][i] - t[n], 2);
	}
	return difference;
}



//逆誤差伝播(逐次)
void backwordS(int n, double studyRate) {
	int i, j, k, l;
	double dLdaSum;
	double d = D[LAYER_NUM + 1];	//平均をとるため出力層の要素数使用
	double dLdy[OUT_NUM], dLda[LAYER_NUM+2][NMAX];

	for (i = LAYER_NUM; i >= 0; i--) {
		//最下層の場合
		if (i == LAYER_NUM) {
			for (k = 0; k < D[i+1]; k++) {
				for (j = 0; j < D[i]; j++) {
					dLda[i + 1][k] = 1 / d * 2 * (z[i + 1][k] - t[n]) * z[i + 1][k] * (1 - z[i + 1][k]);
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

void wInit() {	//重み初期化関数
	random_device rnd;
	mt19937 mt(rnd());
	uniform_real_distribution<> rand12(0.0, 1.0);
	for (int i = 0; i < (LAYER_NUM + 1); ++i) {
		for (int j = 0; j < NMAX; j++) {
			for (int k = 0; k < NMAX; k++) {
				w[i][j][k] = rand12(mt);
			}
		}
	}
}