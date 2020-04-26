#pragma warning(disable: 4996)

#include "header.h"

double x[IN_NUM];
double y[OUT_NUM];
//double w[LAYER_NUM + 1][NMAX][NMAX];
int D[LAYER_NUM + 2] = { 3,4,1 };	//各層のデータ数
//double a[LAYER_NUM + 2][NMAX];	//各層への入力
//double z[LAYER_NUM + 2][NMAX];	//各層からの出力
//double tIn[DATASET_NUM][NMAX];	//教師データ(入力)
//double tOut[DATASET_NUM][NMAX];	//教師データ(出力)

double abc = 0;

int main(void)
{
	int dataN = DATASET_NUM;
	int inputNum = IN_NUM;
	int outputNum = OUT_NUM;
	vector<vector<double>> tIn(dataN, vector<double>(inputNum));  //教師入力x[dataN][inputNum]
	vector<vector<double>> tOut(dataN, vector<double>(outputNum));  //教師出力y[dataN][outputNum]
	vector<vector<double>> disIn(dataN, vector<double>(inputNum));  //識別入力x[dataN][inputNum]
	vector<vector<double>> disOut(dataN, vector<double>(outputNum));  //識別出力y[dataN][outputNum]
	vector<vector<vector<double>>> w;   //w[l][i][j]:l層i番目の出力を(l+1)層j番目の入力に伝達するときの重み
	
	int Flag = TRUE;

	string inputFile = "test_in.csv";
	string outputFile = "test_out.csv";
	string inputDisFile = "dis_in.csv";
	string outputDisFile = "dis_out.csv";
	ifstream ifs(inputFile);
	ifstream ifsT(outputFile);
	ifstream ifsD(inputDisFile);
	ifstream ifsTD(outputDisFile);
	string str;
	int n = 0;

	if (ifs.fail()) {
		cout << "Failed to open " << inputFile << "." << endl;
		return -1;
	}

	if (ifsT.fail()) {
		cout << "Failed to open " << outputFile << "." << endl;
		return -1;
	}

	if (ifsD.fail()) {
		cout << "Failed to open " << inputDisFile << "." << endl;
		return -1;
	}

	if (ifsTD.fail()) {
		cout << "Failed to open " << outputDisFile << "." << endl;
		return -1;
	}

	while (getline(ifs, str)) {
		istringstream line(str);
		string s;
		int d = 0;
		while (getline(line, s, ' ')) {
			double tmp = stod(s);
			tIn[n][d] = tmp;
			d++;
		}
		n++;
	}   //読み込み終了

	//教師出力データ読み込み
	n = 0;
	while (getline(ifsT, str)) {
		istringstream line(str);
		string s;
		int d = 0;
		while (getline(line, s, ' ')) {
			double tmp = stod(s);
			tOut[n][d] = tmp;
			d++;
		}
		n++;
	}   //読み込み終了

	while(getline(ifsT, str)) {
		istringstream line(str);
		string s;
		int d = 0;
		while (getline(line, s, ' ')) {
			double tmp = stod(s);
			disIn[n][d] = tmp;
			d++;
		}
		n++;
	}   //読み込み終了

	//教師出力データ読み込み
	n = 0;
	while (getline(ifsTD, str)) {
		istringstream line(str);
		string s;
		int d = 0;
		while (getline(line, s, ' ')) {
			double tmp = stod(s);
			disOut[n][d] = tmp;
			d++;
		}
		n++;
	}   //読み込み終了

	//学習
	//serial(w, tIn, tOut);
	lump(w, tIn, tOut);

	//結果出力
	vector<vector<vector<double>>> a;		//a[dataN][l][M]:dataN番目データのl層Mノードへの入力値
	vector<vector<vector<double>>> z;		//z[dataN][l][M]:dataN番目データのl層Mノードからの出力値
	a.resize(dataN);	z.resize(dataN);
	for (int n = 0; n < dataN; n++) {
		a[n].resize(LAYER_NUM + 2);	z[n].resize(LAYER_NUM + 2);
		for (int l = 0; l < LAYER_NUM + 2; l++) {
			a[n][l].resize(NMAX);	z[n][l].resize(NMAX);
		}
	}

	for (int n = 0; n < dataN; n++) {
		forward(n, tIn, tOut, a, z, w);
		cout << "#" << n + 1 << "教師データ出力結果：( ";
		for (int k = 0; k < OUT_NUM; k++)
			cout << tOut[n][k] << " ";
		cout << ") → ( ";
		for (int k = 0; k < OUT_NUM; k++) 
			cout << z[n][LAYER_NUM + 1][k] << " ";
		cout << ")" << endl;
	}

	for (int n = 0; n < DIS_DATASET_NUM; n++) {
		forward(n, disIn, disOut, a, z, w);
		cout << "#" << n + 1 << "識別データ出力結果：( ";
		for (int k = 0; k < OUT_NUM; k++)
			cout << disOut[n][k] << " ";
		cout << ") → ( ";
		for (int k = 0; k < OUT_NUM; k++)
			cout << z[n][LAYER_NUM + 1][k] << " ";
		cout << ")" << endl;
	}
	
    return 0;
}