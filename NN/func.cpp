#include "header.h"


//�������֐�(�V�O���C�h�֐�)
double sigmoid(double x) {
	return 1 / (1 + exp(-x));
}

//�����w�K
void serial(double studyRate) {
	int n = 0, count = 0;
	double difference;
	//�܂����ׂẴf�[�^����x�ʂ�(�s�v�Ȃ�Ȃ�)
	for (n = 0; n < DATASET_NUM; n++) {
		forward(n);
		//�t�`�d
		backwordS(n, studyRate);
	}
	while (count < COUNTMAX) {
		n = count % DATASET_NUM;	//0 �� n �� DATASET_NUM - 1
		//���`�d
		difference = forward(n);
		//�덷���������l�����Ȃ珈�����I������
		if (difference < Threshold)
			break;
		//�t�`�d
		backwordS(n, studyRate);
		count++;
	}
	

	return;
}

//�ꊇ�w�K
void lump(double studyRate) {
	int i, j, k, l, n, count = 0;
	double dNum = DATASET_NUM;
	double dLdaSum;
	double d = D[LAYER_NUM + 1];	//���ς��Ƃ邽�ߏo�͑w�̗v�f���g�p
	double dLda[LAYER_NUM + 2][NMAX] = {0};
	double difference;
	//z[0] = x(���͒l)
	while (count < COUNTMAX) {
		count++;
		//���`�d
		difference = 0;
		for (n = 0; n < DATASET_NUM; n++) {
			difference += forward(n);	//n�Ԗڂ̃f�[�^�Z�b�g��ʂ�
			//�f�[�^�Z�b�g��ʂ����т�dL/da���v�Z���A���ꂼ���dL/da�̘a���Ƃ�
			for (i = LAYER_NUM; i >= 0; i--) {
				//�o�͑w�̏ꍇ
				if (i == LAYER_NUM) {					//i:�w�̑��� - 2  (���ԑw�̐�)
					for (k = 0; k < D[i + 1]; k++) {	//0 �� k ��(�o�͑w�̃f�[�^��)
						for (j = 0; j < D[i]; j++) {	//0 �� j ��(�o�͑w�̈�O�̑w�̃f�[�^��)
							dLda[i + 1][k] += 1 / d * 2 * (z[i + 1][k] - tOut[n][k]) * z[i + 1][k] * (1 - z[i + 1][k]);	//dL/da�̘a���Ƃ�
						}
					}
				}
				else {	//�ŉ��w�łȂ��ꍇ
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

		//�덷���������l�����Ȃ珈�����I������
		if (difference < Threshold)
			break;

		//���ׂđ������킹���̂�dLda�͕��ς��Ĉ���(dNum�Ŋ���)
		for (i = LAYER_NUM; i >= 0; i--) {
			//�ŉ��w�̏ꍇ
			if (i == LAYER_NUM) {
				for (k = 0; k < D[i+1]; k++) {
					for (j = 0; j < D[i]; j++) {
						w[i][j][k] -= studyRate * z[i][j] * dLda[i + 1][k] / dNum;
					}
				}
			}
			else {	//�ŉ��w�łȂ��ꍇ
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

//���`�d
double forward(int n) {	//n:�f�[�^�Z�b�g�̔ԍ�
	int i, j, k;
	double difference = 0;
	for (i = 0; i < IN_NUM; i++)
		z[0][i] = tIn[n][i];	//���t�f�[�^��z[0]�ɑ��

	for (i = 1; i < (LAYER_NUM + 2); i++) {
		for (j = 0; j < D[i]; j++) {
			a[i][j] = 0;
			for (k = 0; k < D[i - 1]; k++) {
				a[i][j] += w[i - 1][k][j] * z[i - 1][k];	//�e�w�ւ̓���a��(�d��w*�O�w�̏o��z)�̘a
			}
			z[i][j] = sigmoid(a[i][j]);	//�o�͓͂���a���������֐��ɒʂ�������
		}
	}
	for (i = 0; i < D[LAYER_NUM + 1]; i++) {
		difference += pow(z[LAYER_NUM + 1][i] - tOut[n][i], 2);	//���t�f�[�^�Ƃ̍����̓���Ԃ�
	}
	return difference;
}

//�t�덷�`�d(����)
void backwordS(int n, double studyRate) {
	int i, j, k, l;
	double dLdaSum;
	double d = D[LAYER_NUM + 1];	//���ς��Ƃ邽�ߏo�͑w�̗v�f���g�p
	double dLda[LAYER_NUM+2][NMAX];

	for (i = LAYER_NUM; i >= 0; i--) {
		//�ŉ��w�̏ꍇ
		if (i == LAYER_NUM) {
			for (k = 0; k < D[i+1]; k++) {
				for (j = 0; j < D[i]; j++) {
					dLda[i + 1][k] = 1 / d * 2 * (z[i + 1][k] - tOut[n][k]) * z[i + 1][k] * (1 - z[i + 1][k]);
					w[i][j][k] -=  studyRate * z[i][j] * dLda[i+1][k];
				}
			}
		}
		else {	//�ŉ��w�łȂ��ꍇ
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

//�d�ݏ������֐�
void wInit() {	//�d�ݏ������֐�
	srand(time(NULL));

	for (int i = 0; i < (LAYER_NUM + 1); ++i) {
		for (int j = 0; j < NMAX; j++) {
			for (int k = 0; k < NMAX; k++) {
				w[i][j][k] = (double)rand() / RAND_MAX;
			}
		}
	}
}