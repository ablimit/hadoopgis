#include "TaskMultiply.h"

TaskMultiply::TaskMultiply(int* a, int* b, int* c) {
	this->a = a;
	this->b = b;
	this->c = c;
}

TaskMultiply::~TaskMultiply() {

}

bool TaskMultiply::run(int procType, int tid)
{
	a[0] = b[0] * c[0];

	std::cout << "a[0]="<< a[0] << " b[0]=" << b[0] << " c[0]="<<c[0]<<std::endl;
	// Sleep to make sure that the depencies are being respected.
	sleep(1);

	return true;
}



