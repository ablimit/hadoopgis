/*
 * CallBackComponentExecution.h
 *
 *  Created on: Feb 22, 2012
 *      Author: george
 */

#ifndef CALLBACKCOMPONENTEXECUTION_H_
#define CALLBACKCOMPONENTEXECUTION_H_

#include "Task.h"
#include "Worker.h"

class Worker;

class CallBackComponentExecution: public CallBackTaskBase {
private:
    // Id of the pipeline component associated to this call back
	int componentId;

	// Pointer to Worker process that is executing this call back class/function and the associated tasks
	Worker *worker;

	// Component instance related to this callback
	PipelineComponentBase *compInst;

public:
    CallBackComponentExecution(PipelineComponentBase* compInst, Worker *worker);
    virtual ~CallBackComponentExecution();
    int getComponentId() const;
    void setComponentId(int componentId);
    Worker *getWorker() const;
    void setWorker(Worker *worker);

	bool run(int procType=ExecEngineConstants::GPU, int tid=0);

	PipelineComponentBase* getCompInst() const {
		return compInst;
	}

	void setCompInst(PipelineComponentBase* compInst) {
		this->compInst = compInst;
	}
};

#endif /* CALLBACKCOMPONENTEXECUTION_H_ */
