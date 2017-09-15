
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>

#include "SfMdata.h"
#include "file_ops.h"

#define MAX_LINE_DATA_NUM	4096
#define MAX_LINE_CHARS		4096*10
#define MAX_NUM_OF_P3D		1000000
#define MAX_NUM_OF_CAMS		100000


char line_buff[MAX_LINE_CHARS];
double datas[MAX_LINE_DATA_NUM];


int readDataFromLine(FILE *fp, double *data, int *nDatas);


/*
读取相机文件中的相机内参
======相机文件格式======
#NumOfInsintrics
ax(px)  ay(px)  ppx(px)  ppy(px)  s  quaternion  transl
...
======输入=============
filename	==>相机数据文件
======输出=============
nCams		==>相机(view)的数量
Kdata		==>nCams*9,相机内参数据
initRot		==>nCams*4,相机的初始R,四元数表示
initTrans	==>nCams*3,相机的初始t
======返回值===========
==0表示读取正确，>0表示在那一行出现错误
*/
int readCamsFile(const char *filename, SfMdata &sfmdata)
{
	FILE *fp;
	int nDatas,nRotPara, ret, lineno, camCnt;
	double out_para[12];

	//读取相机文件和点文件
	if (fopen_s(&fp, filename, "r") != 0) {
		fprintf(stderr, "cannot open file %s, exiting\n", filename);
		system("pause");
		exit(1);
	}

	//遍历整个文件，确定有多少个相机
	int nCams = 0;
	lineno = 0;
	while (1) {
		ret = readDataFromLine(fp, datas, &nDatas);
		if (ret != 1) lineno++;
		else break; //文件尾
		if (nDatas == 0) continue;
		if (nDatas != 12 && nDatas !=17) { return lineno - 1; }
		nRotPara = nDatas - 3-5;
		//分析有多少2D点
		(nCams)++;
	}

	sfmdata.initCams(nCams, nRotPara);

	//回到文件开始处
	fpos_t pos = 0;
	fsetpos(fp, &pos);
	//解析相机数据
	camCnt=0;
	while(1) {
		ret = readDataFromLine(fp, datas, &nDatas);
		if(ret!=1) lineno ++;
		else break; //文件尾
		if (nDatas == 0) continue;
		if (nDatas != 12 && nDatas!=17) goto err;
		//解析数据
		if (nDatas == 12) {
			quat2vec(datas, nDatas, out_para); //out_para变为11个值
			memcpy(datas, out_para, 5); memcpy(&datas[6], &out_para[5], 3), memcpy(&datas[5 + nRotPara], &out_para[8], 3);
			datas[5] = sqrt(1.0 - out_para[5] * out_para[5] - out_para[6] * out_para[6] - out_para[7] * out_para[7]);
		}
		sfmdata.add_camera(camCnt, &datas[0], &datas[5], &datas[5+ nRotPara]);
		camCnt++;
	}

	/*
	double* ptr;
	for (int i = 0; i < camCnt; i++)
	{
		printf("=====Cam %d=====\n", i);
		ptr = (Kdata + i * 5);
		printf("K:     %f,%f,%f,%f,%f\n", *ptr, *(ptr + 1), *(ptr + 2), *(ptr + 3), *(ptr + 4));
		ptr = (initRot + i *4);
		printf("Rot:   %f,%f,%f,%f\n", *ptr, *(ptr + 1), *(ptr + 2), *(ptr + 3));
		ptr = (initRot + i * 3);
		printf("Trans: %f,%f,%f\n", *ptr, *(ptr + 1), *(ptr + 2));
	}
	*/
	return 0;
err:
	return lineno - 1;
}


/*
读取点数据，包括3D点，及其对应在view中的投影点
======相机文件格式======
#NumOfPts3D
X Y Z  nframes  frame0 x0 y0  frame1 x1 y1 ...
...
======输入=============
filename	==>点数据文件
======输出=============

======返回值===========
==0 表示读取正确，>0 表示在那一行出现错误
*/
#include <vector>   //查找方便
#include <iostream> 
using namespace std;
int readPtsFile(const char *filename, SfMdata &sfmdata)
{
	int nPts3D, nPts2D, nCams;
	int viewid, p2dCnt, p3dCnt;
	int nDatas, ret, lineno;
	FILE *fp;

	nCams=sfmdata.nCams();
	if (nCams <= 0) return -1;

	//读取相机文件和点文件
	if (fopen_s(&fp, filename, "r") != 0) {
		fprintf(stderr, "cannot open file %s, exiting\n", filename);
		system("pause");
		exit(1);
	}

	//遍历整个文件，确定有多少个3D点和2D点, 确定每个相机包含的3D点的数量，确定每个3D点包含的相机的数量
	nPts2D = 0;
	nPts3D = 0;
	lineno = 0;
	int nPOC = 0, nCOP = 0;
	int *posPoC = new int[nCams+1]; memset(posPoC, 0, nCams * sizeof(int));
	int *posCoP = new int[MAX_NUM_OF_P3D]; memset(posCoP, 0, MAX_NUM_OF_P3D * sizeof(int));
	while (1) {
		ret = readDataFromLine(fp, datas, &nDatas);
		if (ret != 1) lineno++;
		else break; //文件尾
		if (nDatas == 0) continue;
		if ((nDatas - 4) % 3 != 0) goto err;
		if (datas[3] != (nDatas - 4) / 3) goto err;
		//有多少个2D点
		nPts2D = nPts2D + (nDatas - 4) / 3;
		for (int pos = 4; pos < nDatas; pos = pos + 3) {
			//camera id, x,y
			viewid = (int)datas[pos];
			if (viewid < 0 && viewid >= nCams) goto err;
			//更新nP3dOfCams和nCamsOfP3d
			posPoC[viewid]++;
			posCoP[nPts3D]++;
		}
		nPts3D++;
	}
	fpos_t fpos = 0;
	fsetpos(fp, &fpos);

	/*
	cout << "The amount of 3D points for each view:" << endl;
	for (int i = 0; i < nCams; i++)
		cout << posPoC[i] << " ";
	cout << endl;

	cout << "The amount of views for each 3D point:" << endl;
	for (int i = 0; i < nPts3D; i++)
		cout << posCoP[i] << " ";
	cout << endl;
	*/

	//确定每个相机包含的3D点索引在p3dIdxOfCam中的位置, 以及POC的总大小
	int cnt=0;
	int tt, i;
	for (i = 0; i < nCams; i++)	{
		nPOC += posPoC[i];
		if (posPoC[i] == 0) posPoC[i] = -1;
		else {
			tt = posPoC[i];
			posPoC[i] = cnt;
			cnt = tt + cnt;
		}
	}
	posPoC[i] = cnt;
	//确定每个3D点包含的相机索引在camIdxOfP3d中的位置，以及COP的总大小
	cnt = 0;
	for (i = 0; i < nPts3D; i++) {
		nCOP += posCoP[i];
		if (posCoP[i] == 0) posCoP[i] = -1;
		else {
			tt = posCoP[i];
			posCoP[i] = cnt;
			cnt = tt + cnt;
		}
	}
	posCoP[i] = cnt;
	/*
	cout << "The amount of 3D points for each view:" << endl;
	for (int i = 0; i < nCams; i++)
		cout << posPoC[i] << " ";
	cout << endl;

	cout << "The amount of views for each 3D point:" << endl;
	for (int i = 0; i < nPts3D; i++)
		cout << posCoP[i] << " ";
	cout << endl;
	*/

	//为相机数据分配空间
	sfmdata.init(nPts3D, nPts2D, nPOC,nCOP, posPoC,posCoP);
	//sfmdata.display_idxs();

	//解析相机数据
	p3dCnt = 0;
	p2dCnt = 0; //当前读取的2D点计数
	while (1) {
		ret = readDataFromLine(fp, datas, &nDatas);
		if (ret != 1) lineno++;
		else break; //文件尾
		if (nDatas == 0) continue;
		if ((nDatas-4)%3 != 0) goto err;
		if (datas[3] != (nDatas - 4) / 3) goto err;
		//=====解析数据
		//读取3D点，存储到pts3D中
		sfmdata.add_p3d(p3dCnt, (float)datas[0],(float)datas[1],(float)datas[2]);
		//读取对应的2D点
		int p2dCntOfLine = 0;
		for (int pos = 4; pos < nDatas; pos = pos + 3) {
			//camera id, x,y
			viewid = (int)datas[pos];
			//x = datas[pos++]; y = datas[pos++];
			if (viewid < 0 && viewid >= nCams) goto err;
			//存储到pts2D中
			sfmdata.add_p2d(p2dCnt, (float)datas[pos + 1], (float)datas[pos + 2]);
			//更新索引
			sfmdata.set_iidx(p2dCnt, p3dCnt);
			sfmdata.set_jidx(p2dCnt, viewid);
			sfmdata.registerDijPos(p3dCnt, viewid, p2dCnt);
			sfmdata.registerPOCandCOP(p3dCnt, viewid, p2dCnt);
			p2dCnt++;
			p2dCntOfLine++;
		}
		p3dCnt++;
	}

	sfmdata.finishRead();
	//sfmdata.display_idxs();
	//sfmdata.display_pts();
	delete[]posPoC; delete[]posCoP;
	return 0;
err:
	delete[]posPoC; delete[]posCoP;
	return lineno-1;
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
	ret=fgets(line_buff, MAX_LINE_CHARS - 1, fp);
	if (ret == NULL) return 1;
	nChars=(int)strlen(line_buff); //包含'\n'
	while (pos < nChars)
	{
		//跳过多余的分割符
		while ( (line_buff[pos] == '\t' || line_buff[pos] == ',' || line_buff[pos] == ' ') && line_buff[pos] != '\n')
			pos++;
		if (line_buff[pos] == '\n' || line_buff[pos] == '#') break;

		//pos开始的字符可能是一个数字, 进行解析
		sscanf_s(&line_buff[pos], "%lf", &val);
		data[(*nDatas)++] = val;

		//跳过数据，直到遇到分隔符
		while ( line_buff[pos] != '\t' && line_buff[pos] != ',' && line_buff[pos] != ' ' && line_buff[pos] != '\n')
			pos++;
	}

	return 0;
}


/*
输入：
inp		-->包含内参K(5,可选)，畸变(5,可选),旋转（4，四元数），位移（3）
nin		-->输入参数的维度
输出：
outp	-->包含内参K(5,可选),畸变(5,可选),旋转（3，四元数的向量部分）,位移（3）
nout	-->输出参数的维度
*/
void quat2vec(double *inp, int nin, double *outp)
{
	double mag, sg;
	register int i;

	/* intrinsics & distortion */
	if (nin>7) // are they present?
		for (i = 0; i<nin - 7; ++i)
			outp[i] = inp[i];
	else
		i = 0;

	/* rotation */
	/* normalize and ensure that the quaternion's scalar component is non-negative;
	* if not, negate the quaternion since two quaternions q and -q represent the
	* same rotation
	*/
	mag = sqrt(inp[i] * inp[i] + inp[i + 1] * inp[i + 1] + inp[i + 2] * inp[i + 2] + inp[i + 3] * inp[i + 3]);
	sg = (inp[i] >= 0.0) ? 1.0 : -1.0;
	mag = sg / mag;
	outp[i] = inp[i + 1] * mag;
	outp[i + 1] = inp[i + 2] * mag;
	outp[i + 2] = inp[i + 3] * mag;
	i += 3;

	/* translation*/
	for (; i<11; ++i)
		outp[i] = inp[i + 1];
}