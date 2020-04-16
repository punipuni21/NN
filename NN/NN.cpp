#pragma warning(disable: 4996)

#include "header.h"



int main(void)
{
	FILE* fp;

	int Flag = TRUE;
	double studyRate = 0.5;
	int i, j;

	//alternative
	D[0] = 3, D[1] = 4, D[2] = 1;
	
	if ((fp = fopen("test_in.csv", "r")) == NULL) {
		cout << "Input File Open ERROR!" << endl;
		return 0;
	}
	for (i = 0; i < IN_NUM; i++) {
		//if (fscanf(fp, "%lf", &layerOut[0][i]) == EOF) {
		//	break;
		//}
		fscanf(fp, "%lf", &z[0][i]);
	}
	
	

	//wの初期設定
	

	/*while (Flag == TRUE) {

	}
	*/



    return 0;
}

