#include "SfMdata.h"


#define MAX_LINE_DATA_NUM	4096
#define MAX_LINE_CHARS		4096*10
#define MAX_NUM_OF_P3D		1000000
#define MAX_NUM_OF_CAMS		100000
char line_buff[MAX_LINE_CHARS];
double datas[MAX_LINE_DATA_NUM];

int readDataFromLine(FILE *fp, double *data, int *nDatas);


//通过BAL文件初始化该类, BAL格式本身为Rodriguez
bool SfMdata::initWithBAL(string BAL_fileName, bool useQaut)
{
	FILE *fp;
	int nDatas, nRotPara, ret, lineno, camCnt;
	double params[9];

	cout << "Load BAL file..." << endl;
	//读取相机文件和点文件
	if (fopen_s(&fp, BAL_fileName.c_str(), "r") != 0) {
		fprintf(stderr, "cannot open file %s, exiting\n", BAL_fileName.c_str());
		system("pause");
		exit(1);
	}

	//读取参数数量
	ret = readDataFromLine(fp, datas, &nDatas);
	nCams_ = datas[0];  nPts3D_ = datas[1]; nPts2D_ = datas[2];

	//分配空间
	Kdata = new double[nCams_ * 5]; memset(Kdata, 0, nCams_ * 5 * sizeof(int));
	Poses = new double[nCams_*(1 + 3 + 3)]; //focal + rot + transl
	pts3D = new double[nPts3D_ * 3];
	pts2D = new double[nPts2D_ * 2];
	ji_idx = new int[nPts2D_*2];	memset(ji_idx, NON_EXIST_FLAG, nPts2D_*2 * sizeof(int));
	posPoC = new int[nCams_];	memset(posPoC, 0, nCams_+1* sizeof(int));
	posCoP = new int[nPts3D_];	memset(posCoP, 0, nPts3D_+1 * sizeof(int));
	//读取2D点,  以及各相机包含的3D点计数，各3D点对应的相机计数
	for (int k = 0; k < nPts2D_; k++)
	{
		int t = 2 * k;
		ret = readDataFromLine(fp, datas, &nDatas);
		ji_idx[t] = (int)datas[0];	//相机索引
		ji_idx[t+1] = (int)datas[1];
		posPoC[ji_idx[t]]++;
		posCoP[ji_idx[t+1]]++;
		pts2D[t] = datas[2];
		pts2D[t + 1] = datas[3];
	}
	//读取相机参数，前6个为Rodriguez旋转参数+ 位移， 7为焦距(pixel), 8和9为二四阶径向畸变系数
	for (int j = 0; j < nCams_; j++) {
		for (int n = 0; n < 9;n++) {
			ret = readDataFromLine(fp, datas, &nDatas); params[n] = datas[0];
		}
		memcpy(&Poses[j * 7+1], params, 6*sizeof(double));
		Poses[j * 7] = params[6];
		Kdata[j * 5] = params[6]; Kdata[j * 5 + 3] = 1;
	}
	for (int i = 0; i < nPts3D_; i++) {
		ret = readDataFromLine(fp, datas, &nDatas); params[0] = datas[0];
		ret = readDataFromLine(fp, datas, &nDatas); params[1] = datas[0];
		ret = readDataFromLine(fp, datas, &nDatas); params[2] = datas[0];
		memcpy(&pts3D[i * 3], params, 3*sizeof(double));
	}

	
	return true;
}



//通过SBA格式文件初始化该类, SBA格式本身为Qauternion
bool SfMdata::initWithSBA(string CamsFile, string PtsFile, bool useQaut)
{




	return true;
}




/*
从文件的当前行读取nDatas个数据, 数据以'\t' ','或空格分割
======输入==========
fp		==>文件描述符
data	==>提供的MAX_LINE_DATA_NUM个用于存放数据的空间
======输出==========

======返回值========
==1 该行为文件尾， ==0 表示正常行 ==2
*/
int readDataFromLine(FILE *fp, double *data, int *nDatas)
{
	char *ret;
	int nChars, pos, start;
	double val;

	*nDatas = 0;
	start = 0; pos = 0;
	//将该行读到line_buff
	ret = fgets(line_buff, MAX_LINE_CHARS - 1, fp);
	if (ret == NULL) return 1;
	nChars = (int)strlen(line_buff); //包含'\n'
	while (pos < nChars)
	{
		//跳过多余的分割符
		while ((line_buff[pos] == '\t' || line_buff[pos] == ',' || line_buff[pos] == ' ') && line_buff[pos] != '\n')
			pos++;
		if (line_buff[pos] == '\n' || line_buff[pos] == '#') break;

		//pos开始的字符可能是一个数字, 进行解析
		sscanf_s(&line_buff[pos], "%lf", &val);
		data[(*nDatas)++] = val;

		//跳过数据，直到遇到分隔符
		while (line_buff[pos] != '\t' && line_buff[pos] != ',' && line_buff[pos] != ' ' && line_buff[pos] != '\n')
			pos++;
	}

	return 0;
}
