#pragma once

#define __CL_ENABLE_EXCEPTIONS
#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

#include<map>

using namespace std;

class cldev
{
private:
	std::vector<cl::Platform> platforms;
	std::vector<float> pfVersions;			//平台的版本

	int selectedPf;							//选择使用的平台
	std::vector<cl::Device> selectedDevs;	//选择的平台下的设备
	std::vector<int> selDevCUsize;			//选择的设备的CU数量

	cl::Context context;
	cl::Program program;
	std::map<string, cl::Kernel> kernels;
	std::vector<cl::CommandQueue> queues;
	int SVMmode;
public:
	cldev() {};

	int init(bool dispInfo);	//初始化CL设备
	int selectPfWithMostDev(cl_device_type useType, float versionReq, int SVMsupport);	//选择具有最多Device,版本大于等于version的平台作为计算平台
	int createKernels(vector<string> kernelFiles, vector<string> kernelNames);		//从文件创建kernel
	int createKernelsFromStr(const char*str, vector<string> kernelNames);
	int fileToString(const char *filename, std::string& s);
	bool getSVMmode() { return SVMmode; }

	cl::Context get_context() { return context; }
	cl::CommandQueue get_queue(int devIdx) { return queues[devIdx]; }
	int get_CUsize(int devIdx) { return selDevCUsize[devIdx]; }
	int get_prefer_localsize(int devIdx) { return 32; }
	cl::Kernel* get_kernel(string name) {
		if (kernels.find(name) != kernels.end())
			return &kernels[name];
		else
			return NULL;
	}
	void getKernelInfo(string kernelName);

	void test();
};