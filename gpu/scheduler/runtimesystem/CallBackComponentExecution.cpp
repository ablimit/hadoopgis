/*
 * CallBackComponentExecution.cpp
 *
 *  Created on: Feb 22, 2012
 *      Author: george
 */

#include "CallBackComponentExecution.h"

CallBackComponentExecution::CallBackComponentExecution(PipelineComponentBase* compInst, Worker* worker) {
	this->setComponentId(compInst->getId());
	this->setWorker(worker);
	this->setCompInst(compInst);
}

int CallBackComponentExecution::getComponentId() const
{
    return componentId;
}

Worker *CallBackComponentExecution::getWorker() const
{
    return worker;
}

void CallBackComponentExecution::setWorker(Worker *worker)
{
    this->worker = worker;
}

void CallBackComponentExecution::setComponentId(int componentId)
{
    this->componentId = componentId;
}

CallBackComponentExecution::~CallBackComponentExecution() {
	delete compInst;
}

bool CallBackComponentExecution::run(int procType, int tid)
{
	Worker *curWorker = this->getWorker();

	// Add id of the associated component to the list of components computed
	curWorker->addComputedComponent(this->getComponentId());
	std::cout << "Worker: CallBack: Component id="<< this->getComponentId() << " finished!" <<std::endl;

	return true;
}



