#pragma once


#include "clBufferEx.h"
#include "cldev.h"
#include "autodiff.h"

using namespace autodiff;

struct CLAD_info {
	int Nfp;							//number of fixed parameters
	int Nf;								//number of functions
	vector<tuple<int, int, int>> PI;	//partition infomation, [number of tuples, tuple size, non-derivative parameters of tuple]
};

template<typename T>
struct CLAD_data{
	int Ncb;					//number of combined parameters
	clBufferEx<T> FPs;			//buffer of fixed parameters
	clBufferEx<T> Ps;			//buffer of normal parameters
	clBufferEx<int> CBs;		//buffer of combined indexes
	clBufferEx<T> OVs;			//buffer of output values
	clBufferEx<T> ODs;			//buffer of output derivatives，df/dx
};

template<typename T>
struct AD_NODE {
	int id;
	int op;
	int varId1;
	int varId2;
	T val;
	T dval;
};

template<typename T>
struct Compute_Seq {
	vector<int> NodeIds;
	vector<int> NodeOps;
	vector<int> NodeVarId1s;
	vector<int> NodeVarId2s;
	vector<T> NodeVals;
	vector<int> fwIdx;
	vector<int> revIdx;
};

template<typename T>
class CLAD {
	int WI_SIZE;
	int MULTIPLE;
	int Nadp;			//输入到AD的非固定参数个数

	vector<int> inputIds;
	vector<int> outputIds;
	vector<int> NodeIds;
	vector<int> NodeOps;
	vector<int> NodeVarId1s;
	vector<int> NodeVarId2s;
	vector<T> NodeVals;
	vector<int> fwIdx;
	vector<int> revIdx;
	bool initialized;

	clBufferEx<int> partInfoBuf;
	clBufferEx<T> NodeValsBuf;
	clBufferEx<T> NodeDiffsBuf;
	clBufferEx<int> NodeIdsBuf;
	clBufferEx<int> NodeOpsBuf;
	clBufferEx<int> NodeArg1Buf;
	clBufferEx<int> NodeArg2Buf;
	clBufferEx<int> fwIdxBuf;
	clBufferEx<int> revIdxBuf;
	clBufferEx<int> inputsBuf;
	clBufferEx<int> outputsBuf;

	struct CLAD_info info;
	struct CLAD_data<T> data;

private:
	void CLAD::TopoSort();
	string opStr(int op);
public:
	CLAD() { initialized = false; }
	bool init(vector<ADV<T>>& funcs, vector<ADV<T>> &vars);
	void disp_compute_seq();
	bool prepare4cl(cldev &cd, struct CLAD_info &info);
	bool autodiff(cldev &cd, cl::Kernel *kernel, struct CLAD_data<T> &info);
	void disp_result(int idx);		//显示第idx组结果
};

template<typename T>
bool CLAD<T>::init(vector<ADV<T>>& fs, vector<ADV<T>> &xs)
{
	int nodes4CurLayer = 0, nodes4NextLayer = 0;
	set<int> nodeIdset;		//用于避免重复
	set<int> xIds;			//所有自变量的ID
	int maxNodeId = 0;
	if (initialized) return false;

	for (vector<ADV<T>>::iterator iter = xs.begin(); iter < xs.end(); iter++) {	//获取自变量的id
		xIds.insert((*iter)()->id);
		inputIds.push_back((*iter)()->id);
	}

	queue<ADV_Data<T>*> nodes;
	for (vector<ADV<T>>::iterator iter = fs.begin(); iter < fs.end(); iter++) {
		ADV<T> f = *iter;
		//添加一个func节点
		nodes.push(f()); nodes4CurLayer++;
		outputIds.push_back(f()->id);
	}//for (iter = funcs.begin();
	 //==========生成从fs到xs节点计算序列, 广度遍历=================
	while (!nodes.empty()) {
		while (nodes4CurLayer > 0) {//处理该层所有变量
			ADV_Data<T> *node = nodes.front(); nodes.pop(); nodes4CurLayer--;
			//如果nodeIdset中还不存在则添加该节点
			if (nodeIdset.find(node->id) == nodeIdset.end())
			{
				int sid1, sid2;
				nodeIdset.insert(node->id);
				NodeIds.push_back(node->id);
				if (xIds.find(node->id) != xIds.end()) //如果是输入自变量
					NodeOps.push_back(PLACEHOLDER);
				else NodeOps.push_back(node->op);
				if (node->op == CONST_VAL)
					NodeVals.push_back(node->val);
				else
					NodeVals.push_back(0);
				if (maxNodeId < node->id) maxNodeId = node->id;
				//添加该层变量包含的下一层变量
				if (node->var[0] != NULL) {
					sid1 = node->var[0].get()->id;
					NodeVarId1s.push_back(sid1);
					nodes.push(node->var[0].get()); nodes4NextLayer++;
				}
				else {
					NodeVarId1s.push_back(-1); sid1 = -1;
				}
				if (node->var[1] != NULL) {
					sid2 = node->var[1].get()->id;
					NodeVarId2s.push_back(sid2);
					nodes.push(node->var[1].get()); nodes4NextLayer++;
				}
				else {
					NodeVarId2s.push_back(-1); sid2 = -1;
				}
				//printf("[%d(%d): %d,%d] ", node->id,node->op, sid1, sid2);
			}
		}
		nodes4CurLayer = nodes4NextLayer;
		nodes4NextLayer = 0;
	} //while (!nodes.empty())
	  //printf("\n");
	int fwIdxSize = maxNodeId + 1;
	//根据nodeIds生成ForwardIdx
	map<int, int> pos;
	for (int i = 0; i < NodeIds.size(); i++)
		pos.insert(pair<int, int>(NodeIds[i], i)); //id-->pos in Node
	for (int i = 0; i < fwIdxSize; i++) fwIdx.push_back(-1);
	for (map<int, int>::iterator iter = pos.begin(); iter != pos.end(); iter++)
		fwIdx[iter->first] = iter->second;
	//进行拓扑排序, 生成reverse序列
	TopoSort();
	initialized = true;
	return 0;
}


/*
对生成的计算图进行拓扑排序
*/
template<typename T>
void CLAD<T>::TopoSort()
{
	int idx,  id1, id2;
	//
	int N = fwIdx.size();
	vector<int> prevCnt;
	for (int i = 0; i < N; i++) prevCnt.push_back(0);

	for (int i = 0; i < N; i++)
	{
		if (fwIdx[i] < 0) continue;
		idx = fwIdx[i];
		id1 = NodeVarId1s[idx];			//id是id1和id2的前驱
		id2 = NodeVarId2s[idx];
		if (id1 >= 0)	prevCnt[id1]++;
		if (id2 >= 0) prevCnt[id2]++;
	}
	bool  found = false;
	for (int i = 0; i < N; i++)
	{
		found = false;
		//找到没有前驱的节点输出
		for (int j = 0; j < N; j++)
		{
			if (fwIdx[j] < 0)continue;
			if (prevCnt[j] == 0) {
				idx = fwIdx[j];
				revIdx.push_back(idx);
				//将其后继节点的计数减1
				id1 = NodeVarId1s[idx];			//id是id1和id2的前驱
				id2 = NodeVarId2s[idx];			//
				prevCnt[j] = -1;
				if (id1 >= 0)	prevCnt[id1]--;
				if (id2 >= 0) prevCnt[id2]--;
				found = true;
			}
		}
		if (!found)break;
	}
}

template<typename T>
void CLAD<T>::disp_compute_seq()
{
	set<int> inputSet;
	set<int> outputSet;

	for (int i = 0; i < inputIds.size(); i++)
		inputSet.insert(inputIds[i]);
	for (int i = 0; i < outputIds.size(); i++)
		outputSet.insert(outputIds[i]);

	int idx;
	printf("=====Nodes:\n");
	for (int i = 0; i < fwIdx.size(); i++)
	{
		idx = fwIdx[i]; 	if (idx < 0)continue;
		if (inputSet.find(NodeIds[idx]) != inputSet.end())
			printf("N%d[label=\"I%d\",shape=box,style=filled,fillcolor=green,fontsize=20]\n", NodeIds[idx], NodeIds[idx]);
		else if (outputSet.find(NodeIds[idx]) != outputSet.end())
			printf("N%d[label=\"O%d/%s\",shape=doublecircle,style=filled,fillcolor=red,fontsize=20]\n", NodeIds[idx], NodeIds[idx], opStr(NodeOps[idx]).c_str());
		else
			printf("N%d[label=\"%d/%s\",fontsize=20]\n", NodeIds[idx], NodeIds[idx], opStr(NodeOps[idx]).c_str());
	}
	for (int i = 0; i<fwIdx.size(); i++)
	{
		idx = fwIdx[i]; 	if (idx < 0)continue;
		if (NodeVarId1s[idx] >= 0)
			printf("N%d->N%d\n", NodeVarId1s[idx], NodeIds[idx]);
		if (NodeVarId2s[idx] >= 0 && NodeVarId1s[idx] != NodeVarId2s[idx])
			printf("N%d->N%d\n", NodeVarId2s[idx], NodeIds[idx]);
		//printf("[%d(%d,%f):%d,%d]\n", NodeIds[i], NodeOps[i], NodeVals[i], NodeVarId1s[i], NodeVarId2s[i]);
	}
	printf("=====Forward Computation Idx[id -> pos]:\n");
	for (unsigned int i = 0; i< fwIdx.size(); i++)
	{
		printf("[%d->%d] ", i, fwIdx[i]);
	}
	printf("\n");

	printf("=====Reverse Computation Idx[pos in fwIdx]:\n");
	for (unsigned int i = 0; i< revIdx.size(); i++)
	{
		printf("%d ", revIdx[i]);
	}
	printf("\n");

}

/*
为CL做准备，创建buffer
*/
template<typename T>
bool CLAD<T>::prepare4cl(cldev &cd, struct CLAD_info &info)
{
	if (!initialized) return false;
	this->info = info;
	WI_SIZE = cd.get_prefer_localsize(0);
	MULTIPLE =(int)std::ceil(cd.get_prefer_localsize(0)/2);

	int Ns = info.PI.size();
	int Nn = NodeIds.size();
	printf("CUsize=%d\n", cd.get_CUsize(0));

	//创建并填写分区信息 | 合成偏导组元在AD输入中的起始位置, 合成偏导组元在AD输出中的起始位置，各分区在params中的起始位置, 各组单元包含的参数个数, 组单元中非偏导参数个数|*nParts
	int *partInfo = new int[Ns * 5];
	partInfoBuf = clBufferEx<int>(cd.get_context(), cd.get_queue(0), Ns * 5, MODE_COARSE_SVM);
	int pos1 = 0, pos2 = 0;
	Nadp = 0;		//输入到AD的非固定参数个数
	for (int i = 0; i < Ns; i++)
	{
		partInfo[i * 5] = pos1 + std::get<2>(info.PI[i]);
		partInfo[i * 5 + 1] = Nadp;
		partInfo[i * 5 + 2] = pos2;
		partInfo[i * 5 + 3] = std::get<1>(info.PI[i]);	//组单元中参数的数量
		partInfo[i * 5 + 4] = std::get<2>(info.PI[i]);
		pos1 += std::get<1>(info.PI[i]);
		Nadp += std::get<1>(info.PI[i]) - std::get<2>(info.PI[i]);
		pos2 += std::get<0>(info.PI[i])*std::get<1>(info.PI[i]);
	}
	partInfoBuf.write(0, partInfo, Ns * 5);
	delete[]partInfo;

	T* vals = new T[MULTIPLE * cd.get_CUsize(0) * WI_SIZE * Nn];
	NodeValsBuf = clBufferEx<T>(cd.get_context(), cd.get_queue(0), MULTIPLE *cd.get_CUsize(0) * WI_SIZE * Nn, MODE_COARSE_SVM);	 //cd.get_CUsize(0) * WI_SIZE * Nn
	for (int i = 0; i < MULTIPLE * cd.get_CUsize(0) * WI_SIZE; i++)
		memcpy(&vals[i*Nn], &NodeVals[0], Nn*sizeof(T));
	NodeValsBuf.write(0, vals, MULTIPLE * cd.get_CUsize(0) * WI_SIZE * Nn);	//为了将CONST_VAL节点预先写入

	NodeDiffsBuf= clBufferEx<T>(cd.get_context(), cd.get_queue(0), MULTIPLE *cd.get_CUsize(0) * WI_SIZE * Nn, MODE_COARSE_SVM);

	NodeIdsBuf= clBufferEx<int>(cd.get_context(), cd.get_queue(0), Nn, MODE_COARSE_SVM);
	NodeIdsBuf.write(0, &NodeIds[0], Nn);
	NodeOpsBuf=clBufferEx<int>(cd.get_context(), cd.get_queue(0), Nn, MODE_COARSE_SVM);
	NodeOpsBuf.write(0, &NodeOps[0], Nn);
	NodeArg1Buf=clBufferEx<int>(cd.get_context(), cd.get_queue(0), Nn, MODE_COARSE_SVM);
	NodeArg1Buf.write(0, &NodeVarId1s[0], Nn);
	NodeArg2Buf=clBufferEx<int>(cd.get_context(), cd.get_queue(0), Nn, MODE_COARSE_SVM);
	NodeArg2Buf.write(0, &NodeVarId2s[0], Nn);
	fwIdxBuf=clBufferEx<int>(cd.get_context(), cd.get_queue(0), fwIdx.size(), MODE_COARSE_SVM);
	fwIdxBuf.write(0, &fwIdx[0], fwIdx.size());
	revIdxBuf=clBufferEx<int>(cd.get_context(), cd.get_queue(0), Nn, MODE_COARSE_SVM);
	revIdxBuf.write(0, &revIdx[0], revIdx.size());
	inputsBuf=clBufferEx<int>(cd.get_context(), cd.get_queue(0), inputIds.size(), MODE_COARSE_SVM);
	inputsBuf.write(0, &inputIds[0], inputIds.size());
	outputsBuf=clBufferEx<int>(cd.get_context(), cd.get_queue(0), outputIds.size(), MODE_COARSE_SVM);
	outputsBuf.write(0, &outputIds[0], outputIds.size());
	return true;
}


//使用给定的数据对公式进行自动微分
template<typename T>
bool CLAD<T>::autodiff(cldev &cd, cl::Kernel *kernel, struct CLAD_data<T> &data)
{
	this->data = data;
	//==========填写kernel参数信息
	cl_int err;
	int num = 0;
	err = kernel->setArg<int>(num++, cd.get_CUsize(0));		//0
	err = kernel->setArg<int>(num++, MULTIPLE);
	err = kernel->setArg<int>(num++, info.PI.size());
	err = kernel->setArg<int>(num++, info.Nfp);
	err = kernel->setArg<int>(num++, NodeIds.size());
	err = kernel->setArg<int>(num++, fwIdx.size());
	err = kernel->setArg<int>(num++, info.Nf);			//5
	err = kernel->setArg<int>(num++, data.Ncb);
	err = kernel->setArg<int>(num++, Nadp);

	err = partInfoBuf.SetArgForKernel(*kernel, num++);
	err = NodeIdsBuf.SetArgForKernel(*kernel, num++);
	err = NodeOpsBuf.SetArgForKernel(*kernel, num++);		//10
	err = NodeArg1Buf.SetArgForKernel(*kernel, num++);
	err = NodeArg2Buf.SetArgForKernel(*kernel, num++);
	err = NodeValsBuf.SetArgForKernel(*kernel, num++);
	err = NodeDiffsBuf.SetArgForKernel(*kernel, num++);

	err = fwIdxBuf.SetArgForKernel(*kernel, num++);			//15
	err = revIdxBuf.SetArgForKernel(*kernel, num++);
	err = inputsBuf.SetArgForKernel(*kernel, num++);
	err = outputsBuf.SetArgForKernel(*kernel, num++);

	err = data.CBs.SetArgForKernel(*kernel, num++);
	err = data.FPs.SetArgForKernel(*kernel, num++);			//20
	err = data.Ps.SetArgForKernel(*kernel, num++);
	err = data.ODs.SetArgForKernel(*kernel, num++);
	err = data.OVs.SetArgForKernel(*kernel, num++);			//23
	//err = kernel->setArg(num++, sizeof(int)*NodeIds.size(), (void*)NULL);		//申请local memory用于计算序列的id
	//err = kernel->setArg(num++, sizeof(int)*NodeIds.size(), (void*)NULL);		//申请local memory用于计算序列的op
	//err = kernel->setArg(num++, sizeof(int)*NodeIds.size(), (void*)NULL);		//申请local memory用于计算序列的varId1
	//err = kernel->setArg(num++, sizeof(int)*NodeIds.size(), (void*)NULL);		//申请local memory用于计算序列的varId2
	//err = kernel->setArg(num++, sizeof(int)*fwIdx.size(), (void*)NULL);		//申请local memory用于计算序列的fwIdx
	//err = kernel->setArg(num++, sizeof(int)*NodeIds.size(), (void*)NULL);		//申请local memory用于计算序列的revIdxx
																				
	//=========执行kernel
	cd.get_queue(0).enqueueNDRangeKernel(
		*kernel, cl::NullRange, cl::NDRange(data.Ncb),		//globalWorkSize
		cl::NDRange(WI_SIZE), NULL, NULL); //cl::NDRange(WI_SIZE)
	cd.get_queue(0).finish();

	return true;
}

/*
显示第idx组结果
*/
template<typename T>
void CLAD<T>::disp_result(int idx)	
{
	clBufferPtr<double> outdiff_bptr = data.ODs.get_ptr(false);
	double *outdiff_ptr = outdiff_bptr.get();
	outdiff_ptr = outdiff_ptr + idx*Nadp*info.Nf;
	printf("Out Diffs:\n");
	for (int i = 0; i < Nadp*info.Nf; i++) {
		printf("%.5e  ", outdiff_ptr[i]);
	}
	printf("\n");

	clBufferPtr<double> outval_bptr = data.OVs.get_ptr(false);
	double *outval_ptr = outval_bptr.get();
	outval_ptr = outval_ptr + idx*info.Nf;
	printf("Out Vals:\n");
	for (int i = 0; i < info.Nf; i++) {
		printf("%.5e  ", outval_ptr[i]);
	}
	printf("\n");
}

template<typename T>
string CLAD<T>::opStr(int op) 
{
	switch (op)
	{
	case autodiff::PLACEHOLDER:
		return string("PLACEHOLDER");
	case autodiff::CONST_VAL:
		return string("PLACEHOLDER");
	case autodiff::EQUAL:
		return string("=");
	case autodiff::ADD:
		return string("+");
	case autodiff::SUB:
		return string("-");
	case autodiff::MUL:
		return string("*");
	case autodiff::DIV:
		return string("div");
	case autodiff::SIN:
		return string("sin");
	case autodiff::COS:
		return string("cos");
	case autodiff::EXP:
		return string("exp");
	case autodiff::LOGE:
		return string("ln");
	case autodiff::LOG10:
		return string("lg");
	case autodiff::LOG2:
		return string("log2");
	case autodiff::SQRT:
		return string("sqrt");
	case autodiff::NEG:
		return string("-");
	default:
		return string("xxx");
	}
}