#include "header.h"


//�������֐�(�V�O���C�h�֐�)
double sigmoid(double x) {
	return 1 / (1 + exp(-x));
}

//�����w�K
void serial(vector<vector<vector<double>>>& w, vector<vector<double>> tIn, vector<vector<double>> tOut) {
	
	vector<vector<vector<double>>> a;		//a[dataN][l][M]:dataN�Ԗڃf�[�^��l�wM�m�[�h�ւ̓��͒l
	vector<vector<vector<double>>> z;		//z[dataN][l][M]:dataN�Ԗڃf�[�^��l�wM�m�[�h����̏o�͒l
	vector<vector<vector<double>>> zBack;	//zBack[dataN][l][M]:dataN�Ԗڃf�[�^��l�wM�m�[�h����̏o�͒l(�t�`�d�j

	int middleLayerNum = LAYER_NUM;	//���ԑw��
	double studyRate = STUDYRATE;	//�w�K��
	double dataN = DATASET_NUM;		//�f�[�^�Z�b�g��
	int count = 0;					//���s��
	double difference;				//�덷�̘a(�����֐��̘a)
	//z[dataN][0][k] = x(���͒l)

	//a, z�̗̈�m��
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

	//�d�ݏ�����
	wInit(w);

	while (count < COUNTMAX) {
		
		difference = 0;
		//���`�d
		for (int n = 0; n < dataN; n++) {
			difference += forward(n, tIn, tOut, a, z, w);	//n�Ԗڂ̃f�[�^�Z�b�g��ʂ�
			
			if (difference < Threshold)	//�덷���������l�����Ȃ珈�����I������
				break;
			
			//�t�`�d
			for (int l = LAYER_NUM + 1; l > 0; l--) {
				//�o�͑w�̏ꍇ
				if (l == LAYER_NUM + 1) {
					for (int k = 0; k < OUT_NUM; k++) {	//0 �� k ��(�o�͑w�̃f�[�^��)
						for (int j = 0; j < MIDDLE_NUM; j++) {	//0 �� j ��(�o�͑w�̈�O�̑w�̃f�[�^��)
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

				else {	//�ŉ��w�łȂ��ꍇ
					if (l == 1) {	//���͑w�̑O�w�̏ꍇ
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
		cout << "����:" << difference << endl;
	}
	

	return;
}

//�ꊇ�w�K
void lump(vector<vector<vector<double>>>& w, vector<vector<double>> tIn, vector<vector<double>> tOut) {
	
	
	vector<vector<vector<double>>> a;		//a[dataN][l][M]:dataN�Ԗڃf�[�^��l�wM�m�[�h�ւ̓��͒l
	vector<vector<vector<double>>> z;		//z[dataN][l][M]:dataN�Ԗڃf�[�^��l�wM�m�[�h����̏o�͒l
	vector<vector<vector<double>>> zBack;	//zBack[dataN][l][M]:dataN�Ԗڃf�[�^��l�wM�m�[�h����̏o�͒l(�t�`�d�j

	int middleLayerNum = LAYER_NUM;	//���ԑw��
	double studyRate = STUDYRATE;	//�w�K��
	double dataN = DATASET_NUM;		//�f�[�^�Z�b�g��
	int count = 0;					//���s��
	double difference;				//�덷�̘a(�����֐��̘a)
	//z[dataN][0][k] = x(���͒l)
	
	//a, z�̗̈�m��
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

	//�d�ݏ�����
	wInit(w);
	
	while (count < COUNTMAX) {
		count++;
		difference = 0;

		for (int n = 0; n < dataN; n++) {	//�S�f�[�^�Z�b�g����͂���
			//�������`�d
			difference += forward(n, tIn, tOut, a, z, w);	//n�Ԗڂ̃f�[�^�Z�b�g��ʂ�
		}
		cout << "����:" << difference << endl;
		//�덷���������l�����Ȃ珈�����I������
		if (difference < Threshold)
			break;

		for (int l = LAYER_NUM + 1; l > 0; l--) {

			//�o�͑w�̏ꍇ
			if (l == LAYER_NUM + 1) {
				for (int k = 0; k < OUT_NUM; k++) {	//0 �� k ��(�o�͑w�̃f�[�^��)
					for (int j = 0; j < MIDDLE_NUM; j++) {	//0 �� j ��(�o�͑w�̈�O�̑w�̃f�[�^��)
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

			else {	//�ŉ��w�łȂ��ꍇ
				if (l == 1) {	//���͑w�̑O�w�̏ꍇ
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

//���`�d
double forward(int n, vector<vector<double>> tIn, vector<vector<double>> tOut, vector<vector<vector<double>>>& a, vector<vector<vector<double>>>& z, vector<vector<vector<double>>> w) {
	//����
	//n:�f�[�^�Z�b�g�̔ԍ�
	//tIn[n][k]:���t�f�[�^�i���́j= z[n][0][k]
	//a[n][l][k]:n�Ԗڃf�[�^��l�wk�m�[�h�ւ̓��͒l
	//z[n][l][k]:n�Ԗڃf�[�^��l�wk�m�[�h����̏o�͒l
	//w[l][i][j]:l�wi�Ԗڂ̏o�͂�(l+1)�wj�Ԗڂ̓��͂ɓ`�B����Ƃ��̏d��
	int D[LAYER_NUM + 2];
	for (int l = 0; l < LAYER_NUM + 2; l++) {
		if (l == 0)
			D[l] = IN_NUM;
		else if (l == LAYER_NUM + 1)
			D[l] = OUT_NUM;
		else
			D[l] = MIDDLE_NUM;
	}
	//a��0�ŏ�����
	for (int l = 0; l < LAYER_NUM + 2; l++) {
		for (int k = 0; k < NMAX; k++) {
			a[n][l][k] = 0;
		}
	}

	//���͑w
	z[n][0].resize(D[0]);	//�ʂɗv��Ȃ����ǔO�̂���
	for (int k = 0; k < D[0]; k++)
		z[n][0][k] = tIn[n][k];	//���t�f�[�^��z[0][]�ɑ��(0�w����͑w�Ƃ���)

	//���͑w�ȊO
	for (int l = 1; l < (LAYER_NUM + 2); l++) {
		for (int k = 0; k < D[l]; k++) {		//�����~�X���H
			for (int j = 0; j < D[l - 1]; j++) {
				a[n][l][k] += w[l - 1][j][k] * z[n][l - 1][j];	//�e�w�ւ̓���a��(�d��w*�O�w�̏o��z)�̘a
			}
			z[n][l][k] = sigmoid(a[n][l][k]);	//�o�͓͂���a���������֐��ɒʂ�������
		}
	}

	double difference = 0, d = D[LAYER_NUM + 1];

	for (int i = 0; i < D[LAYER_NUM + 1]; i++) {
		difference += pow(z[n][LAYER_NUM + 1][i] - tOut[n][i], 2);	//���t�f�[�^�Ƃ̍����̓���Ԃ�
	}
	return difference / d;	//�o�͌덷�̕���(1�o�͓�����̌덷)��Ԃ�
}

//�d�ݏ������֐�
void wInit(vector<vector<vector<double>>>& w) {	//�d�ݏ������֐�

	//w�̗̈�m��
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