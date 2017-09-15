#pragma once

#include<queue>
#include<set>
#include <memory>

using namespace std;



namespace autodiff {
	enum OpType
	{
		PLACEHOLDER,	//0
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

	/*
	自动微分中的变量，也即自动微分计算图中的节点
	*/
	template<typename T>
	struct ADV_Data {
	public:
		OpType op;			//产生该变量的算符
		shared_ptr<ADV_Data<T>> var[2];		//算符的左右变量
		T val;	//该变量的值
		T dval; //某个输出对该变量求偏导的值
		static int cnt;
		int id;

		ADV_Data() {
			val = 0;
			dval = 0;
		}
	};
	template<typename T> int ADV_Data<T>::cnt = 0;

	//================================================================================================

	//================================================================================================
	template<typename T>
	class ADV {
	public:
		shared_ptr<ADV_Data<T>> ADVptr;
		ADV() {					//在“ADV x"的情况下调用
			ADVptr = shared_ptr<ADV_Data<T>>(new ADV_Data<T>);
			ADVptr->op = PLACEHOLDER;
			ADVptr->var[0] = NULL;	ADVptr->var[1] = NULL;
			ADVptr->id = ADVptr->cnt; ADVptr->cnt++;
		}
		ADV(shared_ptr<ADV_Data<T>> ptr)
		{
			ADVptr = ptr;
		}
		ADV(const ADV &adv)	//拷贝构造函数，在“ADV x=y", 以及”z=x+y"的情况下调用.
		{
			//ADVptr = shared_ptr<ADV<T>>(new ADV<T>);
			//(*this)().op = EQUAL;
			//(*this)().optrR = adv.ADVptr;
			//(*this)().val = adv()->val;
			ADVptr = adv.ADVptr;
		}
		ADV(const T val) {		//在”ADV x=8.8“的情况下调用
			ADVptr = shared_ptr<ADV_Data<T>>(new ADV_Data<T>);
			ADVptr->id = ADVptr->cnt; ADVptr->cnt++;
			ADVptr->val = val;
			ADVptr->op = PLACEHOLDER;
			ADVptr->var[0] = NULL;	ADVptr->var[1] = NULL;
		}
		ADV<T>& operator=(const ADV<T> &rhs)	//等号赋值， ”z=x+y"时先通过拷贝构造函数创建一个临时ADV，再通过等号赋值给z
		{
			if (this == &rhs)
				return *this;
			else {
				ADVptr = rhs.ADVptr;
				return *this;
			}
		}
		ADV<T>& operator=(const T &val)			//等号赋值， ”z=1.8"时先通过拷贝构造函数创建一个临时ADV，再通过等号赋值给z
		{
			ADVptr->val = val;
			return *this;
		}

		ADV_Data<T>*  operator()() const {
			return ADVptr.get();
		}
		//shared_ptr<ADV_Data<T>> get() const { return ADVptr; }
	}; //end of class

	template<typename T>
	ostream& operator<<(ostream &os, const ADV<T> &adv) {
		os << adv()->val;
		return os;
	}

	template<typename T>
	ADV<T> operator+(const ADV<T> &x, const ADV<T> &y) 	//加法，双目运算符
	{
		ADV<T> adv;
		adv()->val = y()->val + x()->val;
		adv()->op = ADD;
		adv()->var[0] = x.ADVptr;	adv()->var[1] = y.ADVptr;
		return adv;
	}

	template<typename T>
	ADV<T> operator+(const ADV<T> &x, const T &y) 	//加法，双目运算符
	{
		ADV<T> adv, adv_y;
		adv_y()->op = CONST_VAL; adv_y()->val = y;
		adv()->val = x()->val + y;
		adv()->op = ADD;
		adv()->var[0] = x.ADVptr;	adv()->var[1] = adv_y.ADVptr;
		return adv;
	}

	template<typename T>
	ADV<T> operator+(const T &x, const ADV<T> &y) 	//加法，双目运算符
	{
		ADV<T> adv, adv_x;
		adv_x()->op = CONST_VAL; adv_x()->val = x;
		adv()->val = x+ y()->val;
		adv()->op = ADD;
		adv()->var[0] = adv_x.ADVptr;	adv()->var[1] = y.ADVptr;
		return adv;
	}

	template<typename T>
	ADV<T> operator-(const ADV<T> &x) 	//加法，双目运算符
	{
		ADV<T> adv;
		adv()->val = -x()->val;
		adv()->op = NEG;
		adv()->var[0] = x.ADVptr;
		return adv;
	}

	template<typename T>
	ADV<T> operator-(const ADV<T> &x, const ADV<T> &y) 	//减法，ADV+ADV, 双目运算符
	{
		ADV<T> adv;
		adv()->val = x()->val - y()->val;
		adv()->op = SUB;
		adv()->var[0] = x.ADVptr;	adv()->var[1] = y.ADVptr;
		return adv;
	}
	template<typename T>
	ADV<T> operator-(const ADV<T> &x, const T &y) 	//减法，ADV-scalar
	{
		ADV<T> adv, adv_y;
		adv_y()->op = CONST_VAL; adv_y()->val = y;
		adv()->val = x()->val - y;
		adv()->op = SUB;
		adv()->var[0] = x.ADVptr;	adv()->var[1] = adv_y.ADVptr;
		return adv;
	}
	template<typename T>
	ADV<T> operator-(const T &x, const ADV<T> &y) 	//减法，scalar+ADV, 双目运算符
	{
		ADV<T> adv, adv_x;
		adv_x()->op = CONST_VAL; adv_x()->val = x;
		adv()->val =x- y()->val;
		adv()->op = SUB;
		adv()->var[0] = adv_x.ADVptr;	adv()->var[1] = y.ADVptr;
		return adv;
	}

	template<typename T>
	ADV<T> operator*(const ADV<T> &x, const ADV<T> &y) 	//乘法，ADV*ADV, 双目运算符
	{
		ADV<T> adv;
		adv()->val = x()->val * y()->val;
		adv()->op = MUL;
		adv()->var[0] = x.ADVptr;	adv()->var[1] = y.ADVptr;
		return adv;
	}

	template<typename T>
	ADV<T> operator*(const ADV<T> &x, const T &y) 	//乘法，ADV*scalar, 双目运算符
	{
		ADV<T> adv,adv_y;		
		adv_y()->op = CONST_VAL; adv_y()->val = y;
		adv()->val = x()->val * y;
		adv()->op = MUL;
		adv()->var[0] = x.ADVptr;	adv()->var[1] = adv_y.ADVptr;
		return adv;
	}

	template<typename T>
	ADV<T> operator*(const T &x, const ADV<T> &y) 	//乘法，scalar*ADV, 双目运算符
	{
		ADV<T> adv, adv_x;
		adv_x()->op = CONST_VAL; adv_x()->val = x;
		adv()->val = x * y()->val;
		adv()->op = MUL;
		adv()->var[0] = adv_x.ADVptr;	adv()->var[1] = y.ADVptr;
		return adv;
	}

	template<typename T>
	ADV<T> operator/(const ADV<T> &x, const ADV<T> &y) 	//除法，ADV/ADV, 双目运算符
	{
		ADV<T> adv;
		adv()->val = x()->val / y()->val;
		adv()->op = DIV;
		adv()->var[0] = x.ADVptr;	adv()->var[1] = y.ADVptr;
		return adv;
	}

	template<typename T>
	ADV<T> operator/(const ADV<T> &x, const T &y) 	//除法，ADV/scalar, 双目运算符
	{
		ADV<T> adv,adv_y;
		adv_y()->op = CONST_VAL; adv_y()->val = y;
		adv()->val = x()->val / y;
		adv()->op = DIV;
		adv()->var[0] = x.ADVptr;	adv()->var[1] = adv_y.ADVptr;
		return adv;
	}

	template<typename T>
	ADV<T> operator/(const T &x, const ADV<T> &y) 	//除法，scalar/ADV, 双目运算符
	{
		ADV<T> adv, adv_x;
		adv_x()->op = CONST_VAL; adv_x()->val = x;
		adv()->val = x / y()->val;
		adv()->op = DIV;
		adv()->var[0] = adv_x.ADVptr;	adv()->var[1] = y.ADVptr;
		return adv;
	}

	template<typename T>
	ADV<T> sin(ADV<T> &x) 	//正弦
	{
		ADV<T> adv;
		adv()->val = std::sin(x()->val);
		adv()->op = SIN;
		adv()->var[0] = x.ADVptr;
		return adv;
	}

	template<typename T>
	ADV<T> cos(ADV<T> &x) 	//正弦
	{
		ADV<T> adv;
		adv()->val = std::cos(x()->val);
		adv()->op = COS;
		adv()->var[0] = x.ADVptr;
		return adv;
	}

	template<typename T>
	ADV<T> exp(ADV<T> &x) 	//正弦
	{
		ADV<T> adv;
		adv()->val = std::exp(x()->val);
		adv()->op = EXP;
		adv()->var[0] = x.ADVptr;
		return adv;
	}

	template<typename T>
	ADV<T> ln(ADV<T> &x) 	//自然对数
	{
		ADV<T> adv;
		adv()->val = std::log(x()->val);
		adv()->op = LOGE;
		adv()->var[0] = x.ADVptr;
		return adv;
	}

	template<typename T>
	ADV<T> sqrt(ADV<T> &x) 	//开方
	{
		ADV<T> adv;
		adv()->val = std::sqrt(x()->val);
		adv()->op = SQRT;
		adv()->var[0] = x.ADVptr;
		return adv;
	}


	template<typename T>
	ADV<T> dot(ADV<T> *a, ADV<T> *b)
	{
		ADV<T> result;
		result = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
		return result;
	}

	/*
	三维向量的叉乘
	*/
	template<typename T>
	void cross(ADV<T> *a, ADV<T> *b, ADV<T> *out)
	{
		out[0] = a[1] * b[2] - a[2] * b[1];
		out[1] = a[2] * b[0] - a[0] * b[2];
		out[2] = a[0] * b[1] - a[1] * b[0];
	}


#define DVAR ADV<double>

}