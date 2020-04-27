#include "header.h"


//活性化関数(シグモイド関数)
double sigmoid(double x) {
	return 1 / (1 + exp(-x));
}

//逐次学習
void serial(vector<vector<vector<double>>>& w, vector<vector<double>> tIn, vector<vector<double>> tOut) {
	
	vector<vector<vector<double>>> a;		//a[dataN][l][M]:dataN番目データのl層Mノードへの入力値
	vector<vector<vector<double>>> z;		//z[dataN][l][M]:dataN番目データのl層Mノードからの出力値
	vector<vector<vector<double>>> zBack;	//zBack[dataN][l][M]:dataN番目データのl層Mノードからの出力値(逆伝播）

	int middleLayerNum = LAYER_NUM;	//中間層数
	double studyRate = STUDYRATE;	//学習率
	double dataN = DATASET_NUM;		//データセット数
	int count = 0;					//試行回数
	double difference;				//誤差の和(損失関数の和)
	//z[dataN][0][k] = x(入力値)

	//a, zの領域確保
	//a[dataN][LAYER][MAX], z[dataN][LAYER][MAX]
	a.resize(dataN);	z.resize(dataN);	zBack.resize(dataN);
	for (int n = 0; n < dataN; n++) {
		a[n].resize(LAYER_NUM + 2);	z[n].resize(LAYER_NUM + 2);	zBack[n].resize(LAYER_NUM + 2);
		for (int l = 0; l < LAYER_NUM + 2; l++) {
			a[n][l].resize(NMAX);	z[n][l].resize(NMAX);	zBack[n][l].resize(NMAX);
			for (int k = 0; k < NMAX; k++)
				zBack[n][l][k] = 0;
		}
	}

	//重み初期化
	wInit(w);

	while (count < COUNTMAX) {
		
		difference = 0;
		//順伝播
		for (int n = 0; n < dataN; n++) {
			difference += forward(n, tIn, tOut, a, z, w);	//n番目のデータセットを通す
			
			if (difference < Threshold)	//誤差がしきい値未満なら処理を終了する
				break;
			
			//逆伝播
			for (int l = LAYER_NUM + 1; l > 0; l--) {
				//出力層の場合
				if (l == LAYER_NUM + 1) {
					for (int k = 0; k < OUT_NUM; k++) {	//0 ≦ k ≦(出力層のデータ数)
						for (int j = 0; j < MIDDLE_NUM; j++) {	//0 ≦ j ≦(出力層の一つ前の層のデータ数)
							zBack[n][l - 1][j] += 2 * (z[n][l][k] - tOut[n][k]) * z[n][l][k] * (1 - z[n][l][k]) * w[l - 1][j][k];
						}
					}

					for (int j = 0; j < MIDDLE_NUM; j++) {
						for (int k = 0; k < OUT_NUM; k++) {
							double sum =  (z[n][l][k] - tOut[n][k]) * z[n][l - 1][j] * z[n][l][k] * (1 - z[n][l][k]);
							w[l - 1][j][k] -= studyRate * sum / dataN;
						}
					}
				}

				else {	//最下層でない場合
					if (l == 1) {	//入力層の前層の場合
						for (int j = 0; j < IN_NUM; j++) {
							for (int k = 0; k < MIDDLE_NUM; k++) {
								zBack[n][l - 1][j] += zBack[n][l][k] * w[l - 1][j][k] * z[n][l][k] * (1 - z[n][l][k]);
							}
						}

						for (int k = 0; k < MIDDLE_NUM; k++) {
							for (int j = 0; j < IN_NUM; j++) {
								double sum = zBack[n][l][k] * z[n][l][k] * z[n][l - 1][j] * (1 - z[n][l][k]);
								w[l - 1][j][k] -= studyRate * sum / dataN;
							}
						}

					}
					else {
						for (int k = 0; k < MIDDLE_NUM; k++) {
							for (int j = 0; j < MIDDLE_NUM; j++) {
								zBack[n][l - 1][j] += zBack[n][l][k] * w[l - 1][j][k] * z[n][l][k] * (1 - z[n][l][k]);
							}
						}

						for (int k = 0; k < MIDDLE_NUM; k++) {
							for (int j = 0; j < MIDDLE_NUM; j++) {
								double sum = zBack[n][l][k] * z[n][l - 1][j] * z[n][l][k] * (1 - z[n][l][k]);
								w[l - 1][j][k] -= studyRate * sum / dataN;
							}
						}
					}
				}
			}

		}
		count++;
		cout << "差分:" << difference << endl;
	}
	

	return;
}

//一括学習
void lump(vector<vector<vector<double>>>& w, vector<vector<double>> tIn, vector<vector<double>> tOut) {
	
	
	vector<vector<vector<double>>> a;		//a[dataN][l][M]:dataN番目データのl層Mノードへの入力値
	vector<vector<vector<double>>> z;		//z[dataN][l][M]:dataN番目データのl層Mノードからの出力値
	vector<vector<vector<double>>> zBack;	//zBack[dataN][l][M]:dataN番目データのl層Mノードからの出力値(逆伝播）

	int middleLayerNum = LAYER_NUM;	//中間層数
	double studyRate = STUDYRATE;	//学習率
	double dataN = DATASET_NUM;		//データセット数
	int count = 0;					//試行回数
	double difference;				//誤差の和(損失関数の和)
	//z[dataN][0][k] = x(入力値)
	
	//a, zの領域確保
	//a[dataN][LAYER][MAX], z[dataN][LAYER][MAX]
	a.resize(dataN);	z.resize(dataN);	zBack.resize(dataN);
	for (int n = 0; n < dataN; n++) {
		a[n].resize(LAYER_NUM + 2);	z[n].resize(LAYER_NUM + 2);	zBack[n].resize(LAYER_NUM + 2);
		for (int l = 0; l < LAYER_NUM + 2; l++) {
			a[n][l].resize(NMAX);	z[n][l].resize(NMAX);	zBack[n][l].resize(NMAX);
			for (int k = 0; k < NMAX; k++)
				zBack[n][l][k] = 0;
		}	
	}

	//重み初期化
	wInit(w);
	
	while (count < COUNTMAX) {
		count++;
		difference = 0;

		for (int n = 0; n < dataN; n++) {	//全データセットを入力する
			//順方向伝播
			difference += forward(n, tIn, tOut, a, z, w);	//n番目のデータセットを通す
		}
		cout << "差分:" << difference << endl;
		//誤差がしきい値未満なら処理を終了する
		if (difference < Threshold)
			break;

		for (int l = LAYER_NUM + 1; l > 0; l--) {

			//出力層の場合
			if (l == LAYER_NUM + 1) {
				for (int k = 0; k < OUT_NUM; k++) {	//0 ≦ k ≦(出力層のデータ数)
					for (int j = 0; j < MIDDLE_NUM; j++) {	//0 ≦ j ≦(出力層の一つ前の層のデータ数)
						for (int n = 0; n < dataN; n++) 
							zBack[n][l - 1][j] += 2 * (z[n][l][k] - tOut[n][k]) * z[n][l][k] * (1 - z[n][l][k]) * w[l - 1][j][k];
					}
				}
		
				for (int j = 0; j < MIDDLE_NUM; j++) {
					for (int k = 0; k < OUT_NUM; k++) {
						double sum = 0;
						for (int n = 0; n < dataN; n++)
							sum += (z[n][l][k] - tOut[n][k]) * z[n][l - 1][j] * z[n][l][k] * (1 - z[n][l][k]);
						w[l - 1][j][k] -= studyRate * sum / dataN;
					}
				}
			}

			else {	//最下層でない場合
				if (l == 1) {	//入力層の前層の場合
					for (int j = 0; j < IN_NUM; j++) {
						for (int k = 0; k < MIDDLE_NUM; k++) {
							for (int n = 0; n < dataN; n++)
								zBack[n][l - 1][j] += zBack[n][l][k] * w[l - 1][j][k] * z[n][l][k] * (1 - z[n][l][k]);
						}
					}

					for (int k = 0; k < MIDDLE_NUM; k++) {
						for (int j = 0; j < IN_NUM; j++) {
							double sum = 0;
							for (int n = 0; n < dataN; n++)
								sum += zBack[n][l][k] * z[n][l][k] * z[n][l - 1][j] * (1 - z[n][l][k]);
							w[l - 1][j][k] -= studyRate * sum / dataN;
						}
					}

				}
				else {
					for (int k = 0; k < MIDDLE_NUM; k++) {
						for (int j = 0; j < MIDDLE_NUM; j++) {
							for (int n = 0; n < dataN; n++)
								zBack[n][l - 1][j] += zBack[n][l][k] * w[l - 1][j][k] * z[n][l][k] * (1 - z[n][l][k]);
						}
					}

					for (int k = 0; k < MIDDLE_NUM; k++) {
						for (int j = 0; j < MIDDLE_NUM; j++) {
							double sum = 0;
							for (int n = 0; n < dataN; n++)
								sum += zBack[n][l][k] * z[n][l - 1][j] * z[n][l][k] * (1 - z[n][l][k]);
							w[l - 1][j][k] -= studyRate * sum / dataN;
						}
					}
				}
			}
		}
	}
	return;
}

//順伝播
double forward(int n, vector<vector<double>> tIn, vector<vector<double>> tOut, vector<vector<vector<double>>>& a, vector<vector<vector<double>>>& z, vector<vector<vector<double>>> w) {
	//引数
	//n:データセットの番号
	//tIn[n][k]:教師データ（入力）= z[n][0][k]
	//a[n][l][k]:n番目データのl層kノードへの入力値
	//z[n][l][k]:n番目データのl層kノードからの出力値
	//w[l][i][j]:l層i番目の出力を(l+1)層j番目の入力に伝達するときの重み
	int D[LAYER_NUM + 2];
	for (int l = 0; l < LAYER_NUM + 2; l++) {
		if (l == 0)
			D[l] = IN_NUM;
		else if (l == LAYER_NUM + 1)
			D[l] = OUT_NUM;
		else
			D[l] = MIDDLE_NUM;
	}
	//aを0で初期化
	for (int l = 0; l < LAYER_NUM + 2; l++) {
		for (int k = 0; k < NMAX; k++) {
			a[n][l][k] = 0;
		}
	}

	//入力層
	z[n][0].resize(D[0]);	//別に要らないけど念のため
	for (int k = 0; k < D[0]; k++)
		z[n][0][k] = tIn[n][k];	//教師データをz[0][]に代入(0層を入力層とする)

	//入力層以外
	for (int l = 1; l < (LAYER_NUM + 2); l++) {
		for (int k = 0; k < D[l]; k++) {		//ここミスか？
			for (int j = 0; j < D[l - 1]; j++) {
				a[n][l][k] += w[l - 1][j][k] * z[n][l - 1][j];	//各層への入力aは(重みw*前層の出力z)の和
			}
			z[n][l][k] = sigmoid(a[n][l][k]);	//出力は入力aを活性化関数に通したもの
		}
	}

	double difference = 0, d = D[LAYER_NUM + 1];

	for (int i = 0; i < D[LAYER_NUM + 1]; i++) {
		difference += pow(z[n][LAYER_NUM + 1][i] - tOut[n][i], 2);	//教師データとの差分の二乗を返す
	}
	return difference / d;	//出力誤差の平均(1出力当たりの誤差)を返す
}

//重み初期化関数
void wInit(vector<vector<vector<double>>>& w) {	//重み初期化関数

	//wの領域確保
	//w[LAYER_NUM+1][MAX][MAX]
	w.resize(LAYER_NUM + 1);
	for (int l = 0; l < LAYER_NUM + 1; l++) {
		w[l].resize(NMAX);
		for (int k = 0; k < NMAX; k++) {
			w[l][k].resize(NMAX);
		}
	}

	random_device rnd;
	mt19937 mt(rnd());
	uniform_real_distribution<> rand(-1.0, 1.0);
	for (int l = 0; l < (LAYER_NUM + 1); l++) {
		for (int i = 0; i < NMAX; i++) {
			for (int j = 0; j < NMAX; j++) {
				w[l][i][j] = rand(mt);
			}
		}
	}
}