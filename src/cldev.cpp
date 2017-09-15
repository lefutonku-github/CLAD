#include "stdafx.h"

#include <iostream>
#include <fstream>
#include <cstdio>
#include <cstdlib>

#include "cldev.h"
#include "clBufferEx.h"

using namespace std;

//初始化CL设备, 查找计算机上的所有openCL平台及设备
int cldev::init(bool dispInfo)
{
	cl::Platform::get(&platforms);
	if (platforms.size() == 0) {
		std::cout << "Platform size 0\n";
		return -1;
	}
	for (int i = 0; i < platforms.size(); i++) {
		cl::STRING_CLASS str;
		//提取版本数值
		platforms[i].getInfo((cl_platform_info)CL_PLATFORM_VERSION, &str);
		float versionNum;
		sscanf_s(str.c_str() + 6, "%f", &versionNum);
		pfVersions.push_back(versionNum);

		if (dispInfo) {
			//*
			cout << "==========Platform " << i << "==========" << endl;
			cout << "Version: " << versionNum << endl;
			platforms[i].getInfo((cl_platform_info)CL_PLATFORM_NAME, &str);
			cout<<"Name: " << str << endl;
			platforms[i].getInfo((cl_platform_info)CL_PLATFORM_PROFILE, &str);
			cout << "Profile: " << str << endl;

			vector<cl::Device> pDevs;
			platforms[i].getDevices(CL_DEVICE_TYPE_CPU | CL_DEVICE_TYPE_GPU, &pDevs);
			for (int j = 0; j < pDevs.size(); j++)
			{
				cout << "=====Device " << j <<"====="<< endl;
				string str;
				pDevs[j].getInfo(CL_DEVICE_NAME, &str);
				cout << "Device Name: " << str << endl;
				cl_device_type type;
				pDevs[j].getInfo(CL_DEVICE_TYPE, &type);
				cout << "Device Type: ";
				if (type== CL_DEVICE_TYPE_CPU)
					cout << "CPU" << endl;
				else if(type == CL_DEVICE_TYPE_GPU)
					cout<< "GPU" << endl;
				//获取设备的local memory size
				cl_ulong lm_size;
				pDevs[j].getInfo(CL_DEVICE_LOCAL_MEM_SIZE, &lm_size);
				cout << "Local mem size:" << lm_size << endl;
			
				//判断设备的SVM支持
				cl_device_svm_capabilities caps;
				pDevs[j].getInfo(CL_DEVICE_SVM_CAPABILITIES,&caps);
				if (caps & CL_DEVICE_SVM_COARSE_GRAIN_BUFFER) {
					cout << "Coarse grain SVM is supported" << endl;
					SVMmode = MODE_COARSE_SVM;
				}
				else {
					cout << "SVM is not supported." << endl;
					SVMmode = MODE_NO_SVM;
				}

			}
			cout << endl;
			//*/
		}
	}
	return 0;
}

/*
versionReq: 版本要求，必须大于等于versionReq
SVMsupport: SVMsupport的支持要求，0无需SVM支持，1=需粗粒度SVM 2=需细粒度SVM
*/
int cldev::selectPfWithMostDev(cl_device_type useType, float versionReq, int SVMsupport)
{
	cl_int err;
	cl_device_svm_capabilities caps;
	cl_command_queue_properties cqps;
	int idx=-1, maxDevs=0;
	vector<cl::Device> pDevs;
	vector<pair<int, cl::Device>> candidates;
	int* devCntForPf = new int[platforms.size()];
	for (int i = 0; i < platforms.size(); i++) {
		devCntForPf[i] = 0;
		//满足版本要求
		if (pfVersions[i]<versionReq) continue;
		try {
			platforms[i].getDevices(useType, &pDevs);
		}
		catch (exception ex) {
			continue;
		}
		//满足SVM要求，以及DEVICE_QUEUE要求
		int svm ;
		bool deviceQueueSupported=false;
		for (int j = 0; j < pDevs.size(); j++) {
			svm = 0;
			pDevs[j].getInfo(CL_DEVICE_SVM_CAPABILITIES, &caps);
			if (caps & CL_DEVICE_SVM_COARSE_GRAIN_BUFFER) svm = 1;
			else{}
			pDevs[j].getInfo(CL_DEVICE_QUEUE_ON_DEVICE_PROPERTIES, &cqps);
			if ( (cqps & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE) && (cqps & CL_QUEUE_PROFILING_ENABLE) )
				deviceQueueSupported=true;
			if (svm >= SVMsupport && deviceQueueSupported) {
				candidates.push_back(make_pair(i, pDevs[j]));
				devCntForPf[i]++;
			}
		}
		if (devCntForPf[i] > maxDevs) {
			maxDevs = devCntForPf[i];
			idx = i;
		}
	}
	delete[] devCntForPf;

	selectedPf = idx;
	if (idx < 0) return -1;
	//填写设备
	for (int i = 0; i < candidates.size(); i++) {
		if (candidates[i].first == selectedPf) {
			selectedDevs.push_back(candidates[i].second);
			cl_int cu_size;
			candidates[i].second.getInfo(CL_DEVICE_MAX_COMPUTE_UNITS, &cu_size);
			selDevCUsize.push_back(cu_size);
			cout << "CU size:" << cu_size << endl;
		}
	}

	if (candidates.size() < 1) {
		cout << "There is no device meet the requirements." << endl;
		return -1;
	}
	cout << "Selected platform " << idx << endl;

	//建立上下文
	cl::Context context_(selectedDevs, NULL,NULL);
	context = context_;

	//为每个设备创建命令队列
	for (int i = 0; i < selectedDevs.size(); i++) {
		cl::CommandQueue queue(context, selectedDevs[i], 0, &err);
		if (err != 0) return -1;
		queues.push_back(queue);

		/* 在设备上创建一个命令队列 */
		cl_queue_properties properties[] = {
			CL_QUEUE_PROPERTIES, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_ON_DEVICE | CL_QUEUE_ON_DEVICE_DEFAULT,
			CL_QUEUE_SIZE,65535,
			0 };
		clCreateCommandQueueWithProperties(context(), selectedDevs[i](), properties, &err);
	}
	return 0;
}


/*
从文件创建kernel
======输出======
==-1 出错 ==0 OK 
*/
int cldev::createKernels(vector<string> kernelFiles, vector<string> kernelNames)
{
	string combFileContent, singFileContent;
	try {
		int err;
		//将各个cl文件内容进行合并
		for (int i = 0; i < kernelFiles.size(); i++)
		{
			//err = fileToString(kernelFiles[i].c_str(), singFileContent); //从文件载入openCL代码
			//if (err != 0) return i;
			combFileContent.append("#include \"");
			combFileContent.append(kernelFiles[i].c_str());
			combFileContent.append("\"\n");
		}

		//编译cl文件内容
		cl::Program::Sources source(1, std::make_pair(combFileContent.c_str(), combFileContent.size()));
		cl::Program program_(context, source, &err);
		program = program_;
		err = program.build(selectedDevs, "-cl-std=CL2.0", NULL,NULL); //

		//创建kernel
		for (int i = 0; i < kernelNames.size();i++)
		{
			cl::Kernel kernel(program, kernelNames[i].c_str(), &err);
			if (err != 0) return -i;
			kernels[kernelNames[i]] = kernel;
		}
	}
	catch (cl::Error err) {
		char *program_log;
		size_t program_size, log_size;
		cout << "Err Code:" << err.err() << endl;
		/* Find size of log and print to std output */
		clGetProgramBuildInfo(program(), selectedDevs[0](), CL_PROGRAM_BUILD_LOG,
			0, NULL, &log_size);
		program_log = (char*)malloc(log_size + 1);
		program_log[log_size] = '\0';
		clGetProgramBuildInfo(program(), selectedDevs[0](), CL_PROGRAM_BUILD_LOG,
			log_size + 1, program_log, NULL);
		printf("%s\n", program_log);
		free(program_log);
		system("pause");
		exit(1);
	}
	return 0;
}


/*
从文件创建kernel
======输出======
==-1 出错 ==0 OK
*/
int cldev::createKernelsFromStr(const char*str, vector<string> kernelNames)
{
	string combFileContent(str);
	try {
		int err;
		//编译cl文件内容
		cl::Program::Sources source(1, std::make_pair(combFileContent.c_str(), combFileContent.size()));
		cl::Program program_(context, source, &err);
		program = program_;
		err = program.build(selectedDevs, "-cl-std=CL2.0", NULL, NULL); //

		//创建kernel
		for (int i = 0; i < kernelNames.size(); i++)
		{
			cl::Kernel kernel(program, kernelNames[i].c_str(), &err);
			if (err != 0) return -i;
			kernels[kernelNames[i]] = kernel;
		}
	}
	catch (cl::Error err) {
		char *program_log;
		size_t program_size, log_size;
		cout << "Err Code:" << err.err() << endl;
		/* Find size of log and print to std output */
		clGetProgramBuildInfo(program(), selectedDevs[0](), CL_PROGRAM_BUILD_LOG,
			0, NULL, &log_size);
		program_log = (char*)malloc(log_size + 1);
		program_log[log_size] = '\0';
		clGetProgramBuildInfo(program(), selectedDevs[0](), CL_PROGRAM_BUILD_LOG,
			log_size + 1, program_log, NULL);
		printf("%s\n", program_log);
		free(program_log);
		system("pause");
		exit(1);
	}
	return 0;
}

/** 读取文件并将其转为字符串 */
int cldev::fileToString(const char *filename, std::string& s)
{
	size_t size;
	char*  str;
	std::fstream f(filename, (std::fstream::in | std::fstream::binary));
	if (f.is_open())
	{
		size_t fileSize;
		f.seekg(0, std::fstream::end);
		size = fileSize = (size_t)f.tellg();
		f.seekg(0, std::fstream::beg);
		str = new char[size + 1];
		if (!str)
		{
			f.close();
			return 0;
		}
		f.read(str, fileSize);
		f.close();
		str[size] = '\0';
		s = str;
		delete[] str;
		return 0;
	}
	cout << "Error: failed to open file\n:" << filename << endl;
	return -1;
}

void cldev::getKernelInfo(string kernelName)
{
	size_t size[3];
	//kernels[kernelName].getWorkGroupInfo(selectedDevs[0], CL_KERNEL_GLOBAL_WORK_SIZE, &size[0]);
	//printf("Max size of global work items:%d,%d,%d", size[0], size[1], size[2]);
	kernels[kernelName].getWorkGroupInfo(selectedDevs[0], CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, &size[0]);
	printf("Prefered size of work group:%d\n", size[0]);
	kernels[kernelName].getWorkGroupInfo(selectedDevs[0], CL_KERNEL_WORK_GROUP_SIZE, &size[0]);
	printf("Max work group:%d\n", size[0]);



}




void cldev::test()
{
	cl_int err;
	float data[5];
	float *x;

	//创建两个SVM
	//创建pts3D, Pts2D, Poses的数据缓存
	map<string, clBufferEx<float>> buffers;
	buffers["test1"]=clBufferEx<float>(context, queues[0], 10, MODE_COARSE_SVM);

	data[0] = 1.23; data[1] = 2.34; data[2] = 3.45; data[3] = 4.56; data[4] = 5.67;
	buffers["test1"].write(0, data, 5);

	cl::Kernel kernel = kernels["test1"];
	buffers["test1"].SetArgForKernel(kernel, 0);

	//在第二个device上调用内核
	cl::Event event;
	try {
		queues[0].enqueueNDRangeKernel(
			kernel,
			cl::NullRange,
			cl::NDRange(4),
			cl::NullRange,
			NULL,
			NULL);
		//event.wait();
		queues[0].finish();
	
		clBufferPtr<float> ptr = buffers["test1"].get_ptr(false);
		float *ptr2=ptr.get();

		buffers["test1"].read(0, data, 5);
	}
	catch (cl::Error err) {
		std::cerr << "ERROR: " << err.what() << "(" << err.err() << ")" << std::endl;
	}

	buffers["test2"] = clBufferEx<float>(context, queues[0], 100, MODE_COARSE_SVM);
}


void cldev_test()
{
	cldev cd;
	cd.init(false);
	if (cd.selectPfWithMostDev(CL_DEVICE_TYPE_ALL, 2.0, 1)) {
		cout << "No devices satisfy the requirement.";
	}

	vector<string> kernelFiles;
	vector<string> kernelNames;
	kernelNames.push_back("test1");		kernelFiles.push_back("E:\\sync_directory\\workspace\\PSBA\\CL_files\\test1.cl");
	kernelNames.push_back("test2");		kernelFiles.push_back("E:\\sync_directory\\workspace\\PSBA\\CL_files\\test2.cl");
	if (cd.createKernels(kernelFiles, kernelNames) != 0)
		cout << "cldev_test(): create Kernel failed" << endl;
	cd.test();
}

