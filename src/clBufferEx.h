#pragma once

#define __CL_ENABLE_EXCEPTIONS
#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

#include<vector>
#include<memory>
#include"mod_shared_ptr.hpp"

#define MODE_NO_SVM			0
#define MODE_COARSE_SVM		1
//#define MODE_FINE_SVM		2

using namespace std;


template<class T>
class clBufferPtr
{
	int mode;
	int size;
	cl::CommandQueue queue;
	cl::Buffer buffer;
	shared_ptr<T> ptr;				//非SVM buffer的指针
	mod_shared_ptr<T> bufferPtr;	//SVM buffer
	bool write;
public:
	clBufferPtr(cl::CommandQueue queue, cl::Buffer buffer, shared_ptr<T> ptr,  int size, bool write) {
		mode = MODE_NO_SVM;
		this->ptr = ptr;
		this->size = size;
		this->queue = queue;
		this->buffer = buffer;
		this->write = write;
	}
	clBufferPtr(cl::CommandQueue queue, T* ptr,  int size, bool write) {
		mode = MODE_COARSE_SVM;
		this->queue = queue;
		this->size = size;
		if (write)
			clEnqueueSVMMap(queue(), CL_TRUE, CL_MAP_WRITE,
				ptr, size * sizeof(T), 0, 0, 0);
		else
			clEnqueueSVMMap(queue(), CL_TRUE, CL_MAP_READ,
				ptr, size * sizeof(T), 0, 0, 0);
		bufferPtr = mod_shared_ptr<T>(ptr);
	}
	~clBufferPtr() {
		if (mode == MODE_NO_SVM) {
			if (write) {
				//将ptr的内容拷贝到buffer
				queue.enqueueWriteBuffer(buffer, CL_TRUE, 0, size * sizeof(T), ptr.get(), 0, 0);
				queue.finish();
			}
		}
		else if (mode == MODE_COARSE_SVM) {
			if (bufferPtr.use_count() == 1 && mode == 1)
				clEnqueueSVMUnmap(queue(), bufferPtr.get(), 0, 0, 0);
		}
	}
	T* get() {
		if (mode == MODE_NO_SVM) return ptr.get();
		else if (mode == MODE_COARSE_SVM) return bufferPtr.get();
		else return NULL;
	}
};


template<class T>
class clBufferEx
{
private:
	cl::Buffer buffer;
	//shared_ptr<T> ptr;				//非SVM buffer输出时使用
	mod_shared_ptr<T> bufferPtr;	//SVM buffer
	int bufferSize;			//buffer的尺寸（实际的字节数还要乘以sizeof(T)）
	int svmMode;		//0=非SVM模式，1=粗粒度SVM模式
	cl::CommandQueue queue;
	cl::Context context;
public:
	clBufferEx() { }
	/*
	clBufferEx(const clBufferEx<T> &buffer) {
		printf("copy func %x\n", this);
		this->buffer=buffer.buffer;
		this->bufferPtr=buffer.bufferPtr;	//SVM buffer
		
		count++;
	} */

	/*
	clBufferEx<T> & clBufferEx<T>::operator = (const clBufferEx<T>& rhs) {
		//printf("operator = %x\n", this);
		this->buffer = rhs.buffer;
		this->bufferPtr = rhs.bufferPtr;	//SVM buffer
		return *this;
	}
	*/
	clBufferEx(cl::Context, cl::CommandQueue queue, int size, int mode);		//创建指定类型的Buffer
	~clBufferEx();

	int write(int pos, T* src, int size);		//往buffer的指定位置写入数据src
	int writeBlocks(int pos, vector<T*> *src, int block_size);
	int read(int pos, T* dst, int size);			//将buffer的指定位置的数据读出到dst
	cl_int SetArgForKernel(cl::Kernel, int num);		//作为kernel的第num个参数
	clBufferPtr<T> get_ptr(bool write);		//返回ptr; 当使用非SVM方式时，必须就近使用。
	int size() { return bufferSize; }
};


template<class T>
clBufferEx<T>::~clBufferEx()
{
	//printf("deconstruct func %x.\t\t", this);
	if (svmMode == MODE_NO_SVM) {
	}
	else {
		if (bufferPtr.use_count() == 1) {
			void *ptr = (void*)bufferPtr.get();
			//printf("free SVM %x\n", ptr);
			clSVMFree(context(), ptr);
		}
	}
}

template<class T>
clBufferEx<T>::clBufferEx(cl::Context context, cl::CommandQueue queue, int size, int mode)		//创建指定类型的Buffer
{
	cl_int err;

	//printf("construct func %x\n", this);
	this->context = context;
	this->queue = queue;
	this->svmMode = mode;
	this->bufferSize = size;
	if (mode == MODE_NO_SVM) {
		buffer = cl::Buffer(context,CL_MEM_READ_WRITE, size * sizeof(T), NULL, &err);
		cl::detail::errHandler(err, "Buffer create error.");
	}
	else {
		try {
			T* ptr = (T*)clSVMAlloc(context(), CL_MEM_READ_WRITE, size * sizeof(T), 0);
			bufferPtr = mod_shared_ptr<T>(ptr);
			//printf("create SVM %x\n", ptr);
		}
		catch (cl::Error err) {
			std::cerr << "ERROR: " << err.what() << "(" << err.err() << ")" << std::endl;
		}
	}
}

//往buffer的指定位置写入数据src
template<class T>
int clBufferEx<T>::write(int pos, T* src, int size)
{
	cl_int err;
	T *ptr, *ptr_org;
	if (svmMode == MODE_NO_SVM) {
		queue.enqueueWriteBuffer(buffer, CL_TRUE, pos*sizeof(T), size * sizeof(T), src, 0, 0);
		queue.finish();
	}
	else if (svmMode == MODE_COARSE_SVM) {
		ptr_org = bufferPtr.get();
		//映射到主机
		err = clEnqueueSVMMap(queue(), CL_TRUE, CL_MAP_WRITE,
			(void*)ptr_org, bufferSize * sizeof(T), 0, 0, 0);
		//操作缓存
		ptr = ptr_org + pos;
		memcpy((void*)ptr, (void*)src, size * sizeof(T));
		//解除映射
		err = clEnqueueSVMUnmap(queue(), (void*)ptr_org, 0, 0, 0);
	}
	return 0;
}

/*
将src指向的一组块依次写入buffer
*/
template<class T>
int clBufferEx<T>::writeBlocks(int pos, vector<T*> *src, int block_size) {
	cl_int err;
	T *ptr, *ptr_org;
	if (svmMode == MODE_NO_SVM) {
		for (int i = 0; i < src->size(); i++) {
			ptr = (*src)[i];
			queue.enqueueWriteBuffer(buffer, CL_TRUE, i*block_size * sizeof(T),
				block_size * sizeof(T), ptr, 0, 0);
			queue.finish();
		}
	}
	else if (svmMode == MODE_COARSE_SVM) {
		ptr_org = bufferPtr.get();
		//映射到主机
		err = clEnqueueSVMMap(queue(), CL_TRUE, CL_MAP_WRITE,
			(void*)ptr_org, bufferSize * sizeof(T), 0, 0, 0);
		//操作缓存
		for (int i = 0; i < src->size(); i++) {
			ptr = ptr_org + i*block_size;
			memcpy((void*)ptr, (void*)(*src)[i], block_size * sizeof(T));
		}
		//解除映射
		err = clEnqueueSVMUnmap(queue(), (void*)ptr_org, 0, 0, 0);
	}
	return 0;
}

//将buffer的指定位置的数据读出到dst
template<class T>
int clBufferEx<T>::read(int pos, T* dst, int size)
{
	cl_int err;
	T* ptr;
	if (svmMode == MODE_NO_SVM) {
	}
	else if (svmMode == MODE_COARSE_SVM) {
		//映射到主机
		err = clEnqueueSVMMap(queue(), CL_TRUE, CL_MAP_READ,
			bufferPtr.get(), size * sizeof(T), 0, 0, 0);
		//操作缓存
		ptr = bufferPtr.get() + pos;
		memcpy((void*)dst, (void*)ptr, size * sizeof(T));
		//解除映射
		err = clEnqueueSVMUnmap(queue(), bufferPtr.get(), 0, 0, 0);
	}

	return 0;
}

template<class T>
cl_int clBufferEx<T>::SetArgForKernel(cl::Kernel kernel, int num)
{
	cl_int err;
	if (svmMode == MODE_NO_SVM)
		kernel.setArg(num, buffer);
	else
		err = clSetKernelArgSVMPointer(kernel(), num, bufferPtr.get());
	return 0;
}

template<class T>
clBufferPtr<T> clBufferEx<T>::get_ptr(bool write)
{
	if (svmMode == MODE_NO_SVM) {
		//用new创建空间, 并使用智能指针
		shared_ptr<T> ptr = shared_ptr<T>(new T[bufferSize], std::default_delete<T[]>());
		//从OpenCL设备端拷贝数据
		queue.enqueueReadBuffer(buffer, CL_TRUE, 0, bufferSize * sizeof(T), ptr.get(), 0, 0);
		queue.finish();
		return clBufferPtr<T>(queue, buffer, ptr, bufferSize, write);
	}
	else{
		return clBufferPtr<T>(queue, bufferPtr.get(), bufferSize, write);
	}
}

