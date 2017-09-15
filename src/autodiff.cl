
enum OpType
{
	PLACEHOLDER,
	CONST_VAL,
	EQUAL,
	ADD,
	SUB,			//4
	MUL,
	DIV,
	SIN,			//7
	COS,
	EXP,
	LOGE,			//10
	LOG10,
	LOG2,
	SQRT,
	NEG
};
void compute_func(int op, global double *NodeVals, int idx, int idx1, int idx2);
void compute_diff(global int*NodeOps, global double *NodeVals, global double *NodeDiffs, int idx, int idx1, int idx2);

/*
前向计算
*/
__kernel void autodiff(
	const int CUsize,
	const int Multiple,
	const int Ns,					//分区(section)数量  checked
	const int Nfp,					//固定参数的数量  checked
	const int Nn,					//计算节点的数量		checked
	const int fwIdxSize,			//forwardIdx数组的尺寸	checked
	const int Nf,					//输出函数的个数		checked
	const int Ncb,					//合成参数的数量		checked
	const int Nadp,					//AD的输入非固定参数数量
	__global int *PI,				//分区(section)信息，|AD输入的各section的首地址，各分区在params中的起始位置, 各组单元包含的参数个数|	checked

	__global int *NodeIds,		//节点的ID
	__global int *NodeOps,		//节点的操作符
	__global int *NodeVarId1s,		//
	__global int *NodeVarId2s,		//
	__global double *NodeVals,		//
	__global double *NodeDiffs,		//

	__global int* fwIdx,		//正向计算序列的索引	checked
	__global int* revIdx,		//
	__global int* inputNodeIds,		//输入节点的IDs	必须与各分区中组单元的参数的和一样, checked
	__global int* funcNodeIds,		//输出节点的IDs, 			checked
	__global int* combIdxs,			//合成索引，取各分区中对应参数合成起来作为自动微分的输入		checked
	__global double *fixedParams,	//固定参数，不输出其微分结果
	__global double * params,		//提供的输入参数		checked
	__global double * outDiffs,		//微分结果输出， 总数量为(nOutputs*nInputs)*nCombIdxs
	__global double * outVals		//前向结果输出， 总数量为nOutputs*nCombIdxs

	//__local int *NodeIds_local,		//反向计算序列
	//__local int *NodeOps_local,		//反向计算序列
	//__local int *NodeVarId1s_local,		//反向计算序列
	//__local int *NodeVarId2s_local,		//反向计算序列
	//__local int *fwIdx_local,
	//__local int *revIdx_local
)
{
	int idx, id, part, Node_base, base, pos, Ntp, adi_pos;
	int n, j;
	int i = get_global_id(0);	//当前处理的自动微分输入项
	int Node_i =  (get_group_id(0) % (Multiple*CUsize))*get_local_size(0) + get_local_id(0);
	Node_base = Node_i*Nn;

	//将全局的Node拷贝到Node_local
	/*
	event_t evt = async_work_group_copy((local int*)NodeIds_local, (global int*)NodeIds, Nn, 0);
	wait_group_events(1, &evt);
	evt = async_work_group_copy((local int*)NodeOps_local, (global int*)NodeOps, Nn, 0);
	wait_group_events(1, &evt);
	evt = async_work_group_copy((local int*)NodeVarId1s_local, (global int*)NodeVarId1s, Nn, 0);
	wait_group_events(1, &evt);
	evt = async_work_group_copy((local int*)NodeVarId2s_local, (global int*)NodeVarId2s, Nn, 0);
	wait_group_events(1, &evt);
	evt = async_work_group_copy((local int*)fwIdx_local, (global int*)fwIdx, fwIdxSize, 0);
	wait_group_events(1, &evt);
	evt = async_work_group_copy((local int*)revIdx_local, (global int*)revIdx, Nn, 0);
	wait_group_events(1, &evt);
	*/

	//==========填入inputNodeIds的初始值
	//1,填入固定参数
	for (n = 0; n < Nfp; n++)	{
		 id = inputNodeIds[n];		//输入的id
		 idx = fwIdx[id];
		 NodeVals[Node_base+idx] = fixedParams[n];
	}
	//2,从各个分区中取对应的组元的参数值
	pos = Nfp;
	for (n = 0; n < Ns; n++) {
		base = PI[5 * n + 2]; //分区n参数在Ps中的的基址
		part = combIdxs[Ns*i + n];	//分区n中取第idx组的参数
		Ntp = PI[5 * n + 3];		//该分区组单元的参数数量
		for (j = 0; j <Ntp; j++) {
			id = inputNodeIds[pos++];
			idx= fwIdx[id];  //fwIdx_local
			NodeVals[Node_base+idx] = params[base + part*Ntp + j];
		}
	}
	//计算函数值
	for (id = 0; id < fwIdxSize; id++) { //fwIdxSize
		idx = fwIdx[id];
		if (idx < 0) continue;
		int idx1 = fwIdx[NodeVarId1s[idx]];
		int idx2 = fwIdx[NodeVarId2s[idx]];
		int op = NodeOps[idx];
		compute_func(op, &NodeVals[Node_base], idx,idx1,idx2);
	}
	//输出函数值结果
	for (n = 0; n < Nf;n++) {
		id = funcNodeIds[n];
		idx = fwIdx[id];
		outVals[i*Nf + n] = NodeVals[Node_base + idx];
	}
	//分别计算每个函数对输入参数的偏导
	base = i*Nadp*Nf;
	for (int f = 0; f < Nf; f++)
	{
		//清除所有节点上的diff值
		for (n = 0; n < Nn; n++)
			NodeDiffs[Node_base+n] = 0;
		//现将该节点的diff置1
		idx = fwIdx[funcNodeIds[f]];
		NodeDiffs[Node_base+idx] = 1;	//要计算的函数节点为导数置1
		for (n = 0; n < Nn; n++) {
			idx = revIdx[n];
			int idx1 = fwIdx[NodeVarId1s[idx]];	//子节点在Node中的索引
			int idx2 = fwIdx[NodeVarId2s[idx]];
			compute_diff(NodeOps, &NodeVals[Node_base], &NodeDiffs[Node_base], idx, idx1, idx2);
		}
		//将结果存入outDiffs
		for (int p = 0; p < Ns; p++) {
			int p5 = 5 * p;
			int Ndp = PI[p5 + 3] - PI[p5 + 4];			//Ndp为该分区中用于求偏导的参数的个数
			pos = base + p*Nf*PI[p5 + 1] + f*Ndp;		//输出的起始位置  p*Nf*PI[5*p+1]
			adi_pos = PI[5+p];
			for (n = 0; n < Ndp; n++)
			{
				id = inputNodeIds[adi_pos + n]; idx = fwIdx[id];
				outDiffs[pos + n] = NodeDiffs[Node_base+idx];
			}
		}
	}
	//barrier(CLK_LOCAL_MEM_FENCE);
	//barrier(CLK_GLOBAL_MEM_FENCE);
} //end of function


/*
计算指定节点的值
*/
inline void compute_func(int op, global double *NodeVals, int idx,int idx1,int idx2)
{
	switch(op)
	{ 
	case ADD:
		NodeVals[idx] = NodeVals[idx1] + NodeVals[idx2];
		break;
	case SUB:
		NodeVals[idx] = NodeVals[idx1] - NodeVals[idx2];
		break;
	case MUL:
		NodeVals[idx] = NodeVals[idx1] * NodeVals[idx2];
		break;
	case DIV:
		NodeVals[idx] = NodeVals[idx1] / NodeVals[idx2];
		break;
	case SIN:			//
		NodeVals[idx] = sin(NodeVals[idx1]);
		break;
	case COS:
		NodeVals[idx] = cos(NodeVals[idx1]);
		break;
	case EXP:
		NodeVals[idx] = exp(NodeVals[idx1]);
		break;
	case LOGE:	//10
		NodeVals[idx] = log(NodeVals[idx1]);
		break;
	case LOG10:
		NodeVals[idx] = log10(NodeVals[idx1]);
		break;
	case LOG2:
		NodeVals[idx] = log2(NodeVals[idx1]);
		break;
	case SQRT:
		NodeVals[idx] = sqrt(NodeVals[idx1]);
		break;
	case NEG:
		NodeVals[idx] = -NodeVals[idx1];
		break;
	default:
		break;
	}
}

/*
计算指定节点对其子节点的偏导
*/
inline void compute_diff(global int*NodeOps, global double *NodeVals, global double *NodeDiffs, int idx, int idx1,int idx2)
{
	switch (NodeOps[idx])
	{
	case ADD:
		NodeDiffs[idx1] += NodeDiffs[idx];
		NodeDiffs[idx2] += NodeDiffs[idx];
		break;
	case SUB:
		NodeDiffs[idx1] += NodeDiffs[idx];
		NodeDiffs[idx2] += -NodeDiffs[idx];
		break;
	case MUL:
		NodeDiffs[idx1] += NodeDiffs[idx] * NodeVals[idx2];
		NodeDiffs[idx2] += NodeDiffs[idx] * NodeVals[idx1];
		break;
	case DIV:
		NodeDiffs[idx1] += NodeDiffs[idx] / NodeVals[idx2];
		NodeDiffs[idx2] += -NodeDiffs[idx]* NodeVals[idx1] /(NodeVals[idx2]* NodeVals[idx2]);
		break;
	case SIN:			//
		NodeDiffs[idx1] += NodeDiffs[idx]*cos(NodeVals[idx1]);
		break;
	case COS:
		NodeDiffs[idx1] += -NodeDiffs[idx]*sin(NodeVals[idx1]);
		break;
	case EXP:
		NodeDiffs[idx1] += NodeDiffs[idx]*exp(NodeVals[idx1]);
		break;
	case LOGE:	//10
		NodeDiffs[idx1] += NodeDiffs[idx]/NodeVals[idx1];
		break;
	case SQRT:
		NodeDiffs[idx1] += NodeDiffs[idx] /(2*sqrt(NodeVals[idx1]));
		break;
	case NEG:
		NodeDiffs[idx1] += -NodeDiffs[idx];
		break;
	}
}
