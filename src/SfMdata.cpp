
#include "SfMdata.h"

SfMdata::~SfMdata() {
	if (nCams_!=0)
	{
		delete[]Kdata;  delete[] Poses;
		//delete[]initRot;
	}

	if (initialized)
	{
		delete []ji_idx; delete []DijL1Blocks;
		delete[]posPoC; delete[]posCoP; delete[]PoCidxs; delete[]CoPidxs;
		delete []pts2D; delete [] pts3D;
	}
	//清除DijL2BlockList中分配的内存
	vector<int*>::iterator iter = DijL2BlockList.begin();
	for (iter = DijL2BlockList.begin(); iter != DijL2BlockList.end(); iter++) {
		delete [](*iter);
	}
	iter = DijL3BlockList.begin();
	for (iter = DijL3BlockList.begin(); iter != DijL3BlockList.end(); iter++) {
		delete[](*iter);
	}
}

int SfMdata::initCams(int nCams, int nRotPara)
{
	if (this->nCams_ != 0) return -1;
	this->nCams_ = nCams;

	Kdata = new double[nCams * 5];
	//initRot = new double[nCams * nRotPara];
	Poses = new double[nCams*(nRotPara-1 + 3)];

	return 0;
}


//初始化存储空间
int SfMdata::init(int nPts3D, int nPts2D, int nPOC, int nCOP, int*posPoC, int* posCoP) {

	if (nCams_ == 0 || initialized) return -1;	//先要读取相机参数

	this->nPts2D_ = nPts2D;
	this->nPts3D_ = nPts3D;
	this->nPoC_ = nPOC;
	this->nCoP_ = nCOP;

	//确定L1_BLOCK_SIZE和L2_BLOCK_SIZE的尺寸
	L3_BLOCK_SIZE = 250;
	L2_BLOCK_SIZE = 250;	//二级索引BLOCK默认为250
	L1_BLOCK_SIZE = nPts3D / (L2_BLOCK_SIZE*L3_BLOCK_SIZE)+1;
	//L1_BLOCK_SIZE = 1024;


	pts3D = new double[nPts3D * 3];
	pts2D = new double[nPts2D * 2];
	ji_idx = new int[nPts2D];	memset(ji_idx, NON_EXIST_FLAG, nPts2D*2 * sizeof(int));
	//jidx = new int[nPts2D];	memset(jidx, NON_EXIST_FLAG, nPts2D * sizeof(int));

	this->posPoC = new int[nCams_+1];  memcpy(this->posPoC, posPoC, (nCams_+1) * sizeof(int));
	this->posCoP = new int[nPts3D+1]; memcpy(this->posCoP, posCoP, (nPts3D+1) * sizeof(int));
	this->cntPoC = new int[nCams_];  memset(this->cntPoC, 0, nCams_ * sizeof(int));
	this->cntCoP = new int[nPts3D]; memset(this->cntCoP, 0, nPts3D * sizeof(int));
	PoCidxs = new int[nPOC];
	PoCidxs2 = new int[nPOC];
	CoPidxs = new int[nCOP];
	CoPidxs2 = new int[nCOP];

	DijL1Blocks = new int[nCams_ * L1_BLOCK_SIZE * sizeof(int)];
	memset(DijL1Blocks, NON_EXIST_FLAG, nCams_ * L1_BLOCK_SIZE * sizeof(int));

	initialized = true;
	return 0;
}

void SfMdata::finishRead() {
	if (initialized) {
		delete[] cntPoC; delete[]cntCoP;
	}
	readFinished = true;
}


/*
1, 将3D点注册到所属的相机(PoCidxs)中. 2, 将相机注册到所属的3D点(CoPidxs)中
pos为2D点索引
*/
int SfMdata::registerPOCandCOP(int i, int j, int idx2D)
{
	int pos;
	pos = posPoC[j] + cntPoC[j];
	PoCidxs[pos] = i;
	PoCidxs2[pos] = idx2D;
	cntPoC[j]++;

	pos = posCoP[i] + cntCoP[i];
	CoPidxs[pos] = j;
	CoPidxs2[pos] = idx2D;
	cntCoP[i]++;

	return 0;
}


//注册Dij的位置，便于快速查找
/*
=====返回值=====
==-1, 内存分配错误 ==0 正确
*/
int SfMdata::registerDijPos(int i, int j, int DijPos)
{
	int L23_SIZE;
	int L1pos,L2pos,L3pos, L2idx,L3idx, TT;
	int *ptr;
	L23_SIZE = L2_BLOCK_SIZE*L3_BLOCK_SIZE;
	L1pos = i / L23_SIZE;		//确定3D点i的L2区间在L1中的位置
	TT = i - L1pos*L23_SIZE;
	L2pos = TT / L3_BLOCK_SIZE;
	L3pos = TT - L2pos*L3_BLOCK_SIZE;

	if (L1pos < 0 || L1pos >= L1_BLOCK_SIZE) return 1;
	if (L2pos < 0 || L2pos >= L2_BLOCK_SIZE) return 1;
	if (L3pos < 0 || L3pos >= L3_BLOCK_SIZE) return 1;
	//确定L2idx
	L2idx = DijL1Blocks[j*L1_BLOCK_SIZE + L1pos];
	if (L2idx == NON_EXIST_FLAG) //表示该区间还没有分配二级索引块
	{
		//申请一块内存，存入DijL2BlockList
		ptr = new int[L2_BLOCK_SIZE]; memset(ptr, NON_EXIST_FLAG, L2_BLOCK_SIZE * sizeof(int));
		if (ptr == NULL) return -1;
		DijL2BlockList.push_back(ptr);
		L2idx= (DijL2BlockList.size() - 1);
		DijL1Blocks[j*L1_BLOCK_SIZE + L1pos] = L2idx;
	}
	//确定L3idx
	L3idx = DijL2BlockList[L2idx][L2pos];
	if (L3idx == NON_EXIST_FLAG)
	{		
		//申请一块内存，存入DijL2BlockList
		ptr = new int[L3_BLOCK_SIZE]; memset(ptr, NON_EXIST_FLAG, L3_BLOCK_SIZE * sizeof(int));
		if (ptr == NULL) return -1;
		DijL3BlockList.push_back(ptr);
		L3idx = (DijL3BlockList.size() - 1);
		DijL2BlockList[L2idx][L2pos] = L3idx;
	}
	DijL3BlockList[L3idx][L3pos] = DijPos;

	return 0;
}

/*
=====返回值=====
>=0 正常的Dij索引位置， <0 没有索引（即Dij的内容为零） 或错误

*/
int SfMdata::getDijPos(int i, int j)
{
	int L23_SIZE;
	int L1pos, L2pos, L3pos, L2idx, L3idx, TT;
	int *ptr;
	L23_SIZE = L2_BLOCK_SIZE*L3_BLOCK_SIZE;
	L1pos = i / L23_SIZE;		//确定3D点i的L2区间在L1中的位置
	TT = i - L1pos*L23_SIZE;
	L2pos = TT / L3_BLOCK_SIZE;
	L3pos = TT - L2pos*L3_BLOCK_SIZE;

	if (L1pos < 0 || L1pos >= L1_BLOCK_SIZE) return 1;
	if (L2pos < 0 || L2pos >= L2_BLOCK_SIZE) return 1;
	if (L3pos < 0 || L3pos >= L3_BLOCK_SIZE) return 1;

	L2idx=DijL1Blocks[j*L1_BLOCK_SIZE + L1pos];
	if ( L2idx == NON_EXIST_FLAG) return -2;
	L3idx = DijL2BlockList[L2idx][L2pos];
	if (L3idx == NON_EXIST_FLAG) return -2;
	return DijL3BlockList[L3idx][L3pos];
}

void SfMdata::display_idxs()
{
	int *ptr;

	cout << "============ij_idx============" << endl;
	for (int i = 0; i < nPts2D_; i++)
		cout <<"["<< ji_idx[2*i] << " "<< ji_idx[2 * i+1]<<"[";
	cout << endl;

	///*
	cout << "============Dij============" << endl;
	cout << "======L1======" << endl;
	for (int i = 0; i < nCams_; i++) {
		cout << "[j " << i << "]: ";
		for (int j = 0; j < L1_BLOCK_SIZE; j++)
			cout << DijL1Blocks[i*L1_BLOCK_SIZE + j] << " ";
		cout << endl;
	}
	cout << "======L2======" << endl;
	for (int idx = 0; idx < DijL2BlockList.size(); idx++) {
		cout << "[Idx " << idx <<"]: ";
		ptr = DijL2BlockList[idx];
		for (int k = 0; k < L2_BLOCK_SIZE; k++)
			cout << ptr[k] << " ";
		cout << endl;
	}

	cout << "======L3======" << endl;
	for (int idx = 0; idx < DijL3BlockList.size(); idx++) {
		cout << "[Idx " << idx << "]: ";
		ptr = DijL3BlockList[idx];
		for (int k = 0; k < L3_BLOCK_SIZE; k++)
			cout << ptr[k] << " ";
		cout << endl;
	}
	//*/

	cout << "============POC============" << endl;
	cout << "======posPoC======" << endl;
	for (int i = 0; i < (nCams_+1); i++)
		cout << posPoC[i] << " ";
	cout << endl;
	cout << "======PoCidxs======" << endl;
	cout << "size:" << nPoC_ << endl;
	for (int i = 0; i < nPoC_; i++)
		cout << PoCidxs[i] << " ";
	cout << endl;
	cout << "======PoCidxs2======" << endl;
	cout << "size:" << nPoC_ << endl;
	for (int i = 0; i < nPoC_; i++)
		cout << PoCidxs2[i] << " ";
	cout << endl;
	cout << "============COP============" << endl;
	cout << "======posCoP======" << endl;
	for (int i = 0; i < (nPts3D_+1); i++)
		cout << posCoP[i] << " ";
	cout << endl;
	cout << "======CoPidxs======" << endl;
	cout << "size:" << nCoP_ << endl;
	for (int i = 0; i < nCoP_; i++)
		cout << CoPidxs[i] << " ";
	cout << endl;
	cout << "======CoPidxs2======" << endl;
	cout << "size:" << nCoP_ << endl;
	for (int i = 0; i < nCoP_; i++)
		cout << CoPidxs2[i] << " ";
	cout << endl;
}

void SfMdata::display_pts(int firstN2D, int firstN3D)
{
	cout << "=====2D points=====" << endl;
	for (int i = 0; i < firstN2D; i++) {
		printf("[%d,%d]:(%le,%le)  ", ji_idx[i*2],ji_idx[i*2+1], pts2D[i * 2], pts2D[i * 2 + 1]);
		if ((i+1) % 5 == 0) cout << endl;
	}
	cout << endl;
	cout << "=====3D points=====" << endl;
	for (int i = 0; i < firstN3D; i++) {
		printf("[%d]:(%le,%le,%le)  ", i, pts3D[i * 3], pts3D[i * 3 + 1], pts3D[i * 3 + 2]);
		if ((i+1) % 5 == 0) cout << endl;
	}
	cout << endl;
}

void SfMdata::display_info()
{
	cout << "============Info============" << endl;
	cout << "Num. of Cams: " << nCams_ << endl;
	cout << "Num. of Pts 3D: " << nPts3D_ << endl;
	cout << "Num. of Pts 2D: " << nPts2D_ << endl;
	cout << "L1_BLOCK_SIZE: " <<L1_BLOCK_SIZE<< endl;
	cout << "L2_BLOCK_SIZE: " << L2_BLOCK_SIZE << endl;
	cout << "L3_BLOCK_SIZE: " << L3_BLOCK_SIZE << endl;
	cout << "nDijL1Blocks:" << nCams_ << endl;
	cout << "nDijL2Blocks:" << DijL2BlockList.size() << endl;
	cout << "nDijL3Blocks:" << DijL3BlockList.size() << endl;
	cout << "Num. of POC: " << nPoC_ << endl;
	cout << "Num. of COP: " << nCoP_ << endl;

}

void SfMdata::display_cameras(int firstNcams)
{
	double *ptr;
	cout << "============Cameras============" << endl;
	cout << "=====K parameters=====" << endl;
	for (int i = 0; i < firstNcams; i++)
	{
		//cout << "==Camera " << i << "==" << endl;
		ptr = (Kdata + i * 5);
		printf("[%d]: %.8e,%.8e,%.8e,%.8e,%.8e\n", i, *ptr, *(ptr + 1), *(ptr + 2), *(ptr + 3), *(ptr + 4));
	}
	/*
	cout << "=====InitRot=====" << endl;
	for (int i = 0; i < nCams_; i++)
	{
		ptr = (initRot + i * 4);
		printf("[%d]: %le,%le,%le,%le\n", i, *ptr, *(ptr + 1), *(ptr + 2), *(ptr + 3));
	}*/
	cout << "=====Poses=====" << endl;
	for (int i = 0; i < firstNcams; i++)
	{
		ptr = (Poses + i *6);
		printf("[%d]: %.8e,%.8e,%.8e,%.8e,%.8e,%.8e\n",
			i, *ptr, *(ptr + 1), *(ptr + 2), *(ptr + 3), *(ptr + 4), *(ptr + 5));
	}
}



void SfMdata::test_DijLx()
{
	bool Dij_passed = true;
	int k;
	for (k = 0; k < nPts2D_; k++) {
	int i = ji_idx[2*k]; int j= ji_idx[2*k+1];
	if (getDijPos(i, j) != k)
	Dij_passed = false;
	}
	if (Dij_passed) cout << "Dij indexing passed." << endl;
	else cout << "Dij indexing failed.";
}