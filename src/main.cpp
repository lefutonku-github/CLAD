
#include "stdafx.h"

#include <time.h>
#include <cmath>
#include <vector>
#include <iostream>
#include <memory>

#include "autodiff.h"
#include "autodiff_cl.h"

#include "cldev.h"
#include "clBufferEx.h"
#include "SfMdata.h"

using namespace std;
using namespace autodiff;

void test1(cldev &cd);
void BAL_test(cldev &cd);
void BAL_formula(vector<DVAR> &fs, vector<DVAR> &xs);
int main()
{

	//==========openCL initialization
	cldev cd;
	cd.init(true);
	//selecte devices
	if (cd.selectPfWithMostDev(CL_DEVICE_TYPE_CPU, 1.2, 1)) {
		cout << "No devices satisfy the requirement.";
	}
	//==========create kernel
	vector<string> kernelFiles;
	vector<string> kernelNames;
	kernelNames.push_back("autodiff");	kernelFiles.push_back("E:\\sync_directory\\workspace\\AutoDiff\\src\\autodiff.cl");
	if (cd.createKernels(kernelFiles, kernelNames) != 0)
		return 1;

	cd.getKernelInfo("autodiff");


	BAL_test(cd);
	//test1(cd);

	system("pause");
	return 0;
}

void test1(cldev &cd)
{

	//==========construct formula
	DVAR x1 = 2;
	DVAR x2 = 5;

	DVAR sinx2 = sin(x2);
	DVAR lnx1 = ln(x1);

	//x1 = x1 + 3.0;	//实际上会新建一个节点赋值为x1
	DVAR f = lnx1 +sinx2-x2*x2;	// lnx1 + x1*x2 - sinx2;
	DVAR g = lnx1*sinx2;

	vector<DVAR> vars;
	vars.push_back(x1);
	vars.push_back(x2);

	vector<DVAR> funcs;
	funcs.push_back(f);
	funcs.push_back(g);

	//==========fill CLAD_info struct
	CLAD_info info;
	info.Nfp = 0;
	info.Nf = 2;
	info.PI.push_back(tuple<int, int, int>(2, 1, 0));	//第一个分区中的组单元个数，组单元中参数个数
	info.PI.push_back(tuple<int, int, int>(2, 1, 0));

	CLAD<double> clad;
	clad.init(funcs, vars);
	clad.disp_compute_seq();
	clad.prepare4cl(cd, info);

	CLAD_data<double> data;
	data.Ncb = 2;
	//info.Ntp = 4;
	double params[4] = { 2,3,5,6 };			//[1,2 | 8,9]分为两个分区。(1,8), (2,9)形成两组参数, combIdxs为(1,1),(2,2)
	int combIdxs[4] = { 0,0,1,1 };
	data.Ps = clBufferEx<double>(cd.get_context(), cd.get_queue(0), 4, MODE_COARSE_SVM);
	data.Ps.write(0, params, 4);
	data.CBs = clBufferEx<int>(cd.get_context(), cd.get_queue(0), 4, MODE_COARSE_SVM);
	data.CBs.write(0, combIdxs, 4);
	data.FPs = clBufferEx<double>(cd.get_context(), cd.get_queue(0), info.Nfp, MODE_COARSE_SVM);
	data.ODs = clBufferEx<double>(cd.get_context(), cd.get_queue(0), data.Ncb*info.Nf*(1 + 1), MODE_COARSE_SVM);
	data.OVs = clBufferEx<double>(cd.get_context(), cd.get_queue(0), data.Ncb*info.Nf, MODE_COARSE_SVM);

	for (int i = 0; i < 100;i++) {
		clad.autodiff(cd, cd.get_kernel("autodiff"), data);
		int idx = 0;
		clad.disp_result(idx);
	}

	//=========
	printf("f=%.5e\n", f()->val);
	printf("g=%.5e\n", g()->val);

}



//Bundle adjustment testing
void BAL_test(cldev &cd)
{
	//load BAL　data
	SfMdata sfm;
	sfm.initWithBAL("E:\\local_workspace\\PSBA_data\\Ladybug-1723-156502-pre.txt", false);
	sfm.display_cameras(10);
	sfm.display_pts(10, 10);
	clBufferEx<double> Kbuffer = clBufferEx<double>(cd.get_context(), cd.get_queue(0), 5, MODE_COARSE_SVM);
	Kbuffer.write(0, sfm.K_ptr(), 5);
	clBufferEx<double> paramBuffer = clBufferEx<double>(cd.get_context(), cd.get_queue(0), sfm.nCams() * 7 + sfm.nPts3D() * 3, MODE_COARSE_SVM);
	paramBuffer.write(0, sfm.poses_ptr(), sfm.nCams() * 7);
	paramBuffer.write(sfm.nCams() * 7, sfm.pts3D_ptr(), sfm.nPts3D() * 3);

	//sure that the size of buffer is integer multiple of 4,8,16,...
	int N = sfm.nPts2D() / cd.get_prefer_localsize(0)+1;
	N = N*cd.get_prefer_localsize(0);

	clBufferEx<int> ijIdxBuffer = clBufferEx<int>(cd.get_context(), cd.get_queue(0), N*2, MODE_COARSE_SVM);
	ijIdxBuffer.write(0, sfm.ji_idx_ptr(), sfm.nPts2D() * 2);
	clBufferEx<double> jacBuffer = clBufferEx<double>(cd.get_context(), cd.get_queue(0), N*(2 * 6 + 2 * 3), MODE_COARSE_SVM);
	clBufferEx<double> projBuffer = clBufferEx<double>(cd.get_context(), cd.get_queue(0), N * 3, MODE_COARSE_SVM);

	//init CLAD class
	CLAD<double> clad;
	vector<DVAR> fs, xs;
	BAL_formula(fs, xs);
	clad.init(fs, xs);	//generate computation sequence
	//clad.disp_compute_seq();
	//fill CLAD_info struct
	CLAD_info info;
	info.Nf = 2; info.Nfp = 0;
	info.PI.push_back(tuple<int, int, int>(sfm.nCams(), 7, 1));		//[number of tuples, tuple size, non-derivative parameters of tuple]
	info.PI.push_back(tuple<int, int, int>(sfm.nPts3D(), 3, 0));	//
	clad.prepare4cl(cd, info);

	CLAD_data<double> data;
	data.Ncb = N;	//size of combined parameters
	data.Ps = paramBuffer;
	data.CBs = ijIdxBuffer;
	data.ODs = jacBuffer;		//output derivatives
	data.OVs = projBuffer;		//output values of function

	time_t curTm = clock();
	for (int i = 0; i < 100; i++) {
		printf("\b\b\b\b%04d", i);
		clad.autodiff(cd, cd.get_kernel("autodiff"), data);
		//int idx =sfm.nPts2D()-1;
		//clad.disp_result(idx);
		//double *pt = sfm.get_pt(idx);
		//printf("pt2D:%f,%f\n", pt[0], pt[1]);
		//_sleep(1);
	}
	printf("\n");

	time_t tmLast = clock() - curTm;
	printf("nCams=%d, nPts3D=%d, nPts2D=%d\n", sfm.nCams(), sfm.nPts3D(), sfm.nPts2D());
	printf("Time last for Jacobian computation:%d\n", tmLast);

	int idx = sfm.nPts2D() - 1;
	clad.disp_result(idx);
	double *pt = sfm.get_pt(idx);
	printf("pt2D:%f,%f\n", pt[0], pt[1]);

}


void BAL_formula(vector<DVAR> &fs, vector<DVAR> &xs)
{
	//==========construct the formula of projection
	DVAR focal;
	DVAR angle_axis[3];		// = { 1.0,1.0,1.0 };		//rodrigues rotation parameters
	DVAR transl[3];			//
	DVAR pt3D[3];			// = { 1.0,2.0,3.0 };		//3
	DVAR x, y;			//
	//rotating points
	DVAR theta2 = dot<double>(angle_axis, angle_axis);
	DVAR theta = sqrt(theta2);
	DVAR costheta = cos(theta);
	DVAR sintheta = sin(theta);
	DVAR theta_inverse = 1.0 / theta;
	DVAR w[3] = { angle_axis[0] * theta_inverse,
		angle_axis[1] * theta_inverse,
		angle_axis[2] * theta_inverse };
	DVAR w_cross_pt[3];
	cross(w, pt3D, w_cross_pt);
	DVAR tmp = dot(w, pt3D)*(1.0 - costheta);
	DVAR tmp3D[3];
	tmp3D[0] = pt3D[0] * costheta + w_cross_pt[0] * sintheta + w[0] * tmp;
	tmp3D[1] = pt3D[1] * costheta + w_cross_pt[1] * sintheta + w[1] * tmp;
	tmp3D[2] = pt3D[2] * costheta + w_cross_pt[2] * sintheta + w[2] * tmp;
	//translating
	tmp3D[0] = tmp3D[0] + transl[0];
	tmp3D[1] = tmp3D[1] + transl[1];
	tmp3D[2] = tmp3D[2] + transl[2];
	x = -tmp3D[0] / tmp3D[2];
	y = -tmp3D[1] / tmp3D[2];
	x = x*focal;
	y = y*focal;

	//vector<DVAR> vars;
	xs.push_back(focal);
	xs.push_back(angle_axis[0]);
	xs.push_back(angle_axis[1]);
	xs.push_back(angle_axis[2]);
	xs.push_back(transl[0]);
	xs.push_back(transl[1]);
	xs.push_back(transl[2]);
	xs.push_back(pt3D[0]);
	xs.push_back(pt3D[1]);
	xs.push_back(pt3D[2]);
	//vector<DVAR> funcs;
	fs.push_back(x);
	fs.push_back(y);
}
