#pragma once

#include <vector>
#include <iostream>
#include <memory>


/*
每个相机有一个大小为L1_BLOCK_SIZE的一级索引, 其第k个元素如果为零，表示编号为[k*1000, (k+1)*1000)区间的3D点在该相机上没有投影。
如果不为零，则其值l用于指向二级索引块中第l个块，表示该块对应的[k*1000, (k+1)*1000)区间有在该相机上投影的3D点。
*/

#define NON_EXIST_FLAG		-1

using namespace std;
class SfMdata		//该类应该只允许产生一个实例，singleton
{
private:
	//============For cameras=========//
	double *Kdata;		//相机的insintric parameters
	//double *initRot;	//初始的旋转，四元数(4*1)或旋转矩阵(9x1)
	double *Poses;		//相机的Pose, 3+3。初始旋转量+初始位移。 
	bool isQaut;		//指示旋转量是否为Qauternion。否则为Rodriguez形式

	//============For points=========//
	int L1_BLOCK_SIZE;
	int L2_BLOCK_SIZE;
	int L3_BLOCK_SIZE;
	int nCams_;			//相机的数量
	int nPts2D_;			//2D点的数量
	int nPts3D_;			//3D点的数量
	int nPoC_;			//PoCidxs索引的总数量
	int nCoP_;			//CoPidxs索引的总数量

	double *pts2D;		//nPts2D * 2, 2D点数据
	double *pts3D;		//nPts3D * 3, 3D点数据
	int *ji_idx;			//nPts2D * 1, 2D点对应的3D点索引, 注意2D点是按3D点在各相机中的顺序排列的
	//int *jidx;			//nPts2D * 1, 2D点对应的相机索引

	int *PoCidxs;		//记录每个相机包含的3D点索引
	int *PoCidxs2;		//记录每个相机包含的3D点对应的2D点的索引
	int *posPoC;		//nCams*1, 记录每个相机包含的3D点索引在p3dIdxOfCam中的位置
	int *CoPidxs;		//记录每个3D点包含的相机索引
	int *CoPidxs2;		//记录每个3D点包含的相机对应的2D点的索引
	int *posCoP;		//nPts3D*1, 记录每个3D点包含的相机索引在camIdxOfP3d中的位置
	int *cntPoC;		//
	int *cntCoP;

	//用于确定Dij的位置
	int *DijL1Blocks;				//nCams* L1_BLOCK_SIZE, Dij的第一级索引, 
	vector<int *> DijL2BlockList;	//Dij的第二级索引
	vector<int *> DijL3BlockList;	//Dij的第三级索引

	bool initialized;
	bool readFinished;
public:
	~SfMdata();
	SfMdata() {
		initialized = false; readFinished = false;
		nCams_ = 0;
	}

	int initCams(int nCams, int nRotPara);

	int init(int nPts3D, int nPts2D, int nPOC,int nCOP, int*posPoC, int* posCoP);

	void finishRead();

	int registerDijPos(int i, int j, int DijPos);

	int registerPOCandCOP(int i, int j, int idx2D);

	int getDijPos(int i, int j);
	void display_idxs();
	void display_pts(int firstN2D, int firstN3D);
	void display_info();
	void display_cameras(int firstNcams);
	
	void set_iidx(int pos, int i, int j) {
		if (initialized) {
			ji_idx[2 * pos] = i;
			ji_idx[2 * pos + 1] = j;
		}
	}

	int get_iidx(int pos) {
		if (initialized) return ji_idx[2*pos];
		else return -1;
	}

	int get_jidx(int pos) {
		if (initialized) return ji_idx[2*pos+1];
		else return -1;
	}

	void add_p3d(int pos, float X, float Y, float Z) {
		pts3D[pos * 3] = X;
		pts3D[pos * 3 + 1] = Y;
		pts3D[pos * 3 + 2] = Z;
	}

	void add_p2d(int pos, float x, float y) {
		pts2D[pos * 2] = x;
		pts2D[pos * 2 + 1] = y;
	}
	
	double* get_pt(int idx) { return &pts2D[2 * idx]; }

	int nPts3D() { return nPts3D_; }
	int nPts2D() { return nPts2D_; }
	int nCams() { return nCams_; }
	double* pts3D_ptr() { return pts3D; }
	double* pts2D_ptr() { return pts2D; }
	double* poses_ptr() { return Poses; }
	double* K_ptr() { return Kdata; }
	//double* initRot_ptr() { return initRot; }

	int* ji_idx_ptr() { return ji_idx; }
	int* PoCidxs_ptr() { return PoCidxs; }
	int* PoCidxs2_ptr() { return PoCidxs2; }
	int* posPoC_ptr() { return posPoC; }
	int* CoPidxs_ptr() { return CoPidxs; }
	int* CoPidxs2_ptr() { return CoPidxs2; }
	int* posCoP_ptr() { return posCoP;  }
	int nPOC() { return nPoC_; }
	int nCOP() { return nCoP_; }

	int* DijL1Blocks_ptr() { return DijL1Blocks; }
	int DijL1_BLOCK_SIZE() { return L1_BLOCK_SIZE; }
	vector<int*>* DijL2Blocks(){ return &DijL2BlockList; }
	int DijL2_BLOCK_SIZE() { return L2_BLOCK_SIZE; }
	vector<int*>* DijL3Blocks() { return &DijL3BlockList; }
	int DijL3_BLOCK_SIZE() { return L3_BLOCK_SIZE; }

	void test_DijLx();

	//通过BAL文件初始化该类, BAL格式本身为Rodriguez
	bool initWithBAL(string BAL_fileName, bool useQaut);

	//通过SBA格式文件初始化该类, SBA格式本身为Qauternion
	bool initWithSBA(string CamsFile, string PtsFile, bool useQaut);

}; //end of class;