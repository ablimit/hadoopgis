/*
 * ThreadPool.h
 *
 *  Created on: Aug 18, 2011
 *      Author: george
 */

#ifndef THREADPOOL_H_
#define THREADPOOL_H_

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <pthread.h>
#include <semaphore.h>
#include <sys/time.h>

#include "TasksQueue.h"
#include "ExecutionEngine.h"
//#include "ExecEngineConstants.h"

class TasksQueue;
class ExecutionEngine;

struct threadData{
	int tid;
	int procType;
	void *threadPoolPtr;
};

class ThreadPool {
private:
	// list of task from what the thread pool will consume
	TasksQueue *tasksQueue;

	// Is Execution done? Meaning that no more tasks will be inserted for computation within the system?
	bool execDone;

	// Pointer to the execution engine that instantiated this ThreadPool
	ExecutionEngine *execEngine;

	// structure containing information about the threads used to GPU and CPU
	pthread_t *CPUWorkerThreads;
	pthread_t *GPUWorkerThreads;

	int gpuTempDataSize;
	std::vector<void *>gpuTempData;

	int numGPUThreads;
	int numCPUThreads;

	// This mutex is used to prevent the worker threads from initialized after
	// their creation, but only when initExecution function is called.
	pthread_mutex_t initExecutionMutex;

	// Used to make sure that threads are created before the main thread has leaved
	pthread_mutex_t createdThreads;

	// Auxiliar variables used to measured load imbalance in the threads execution time.
	bool firstToFinish;
	struct timeval firstToFinishTime;
	struct timeval lastToFinishTime;

public:
	ThreadPool(TasksQueue *tasksQueues, ExecutionEngine *execEngine);
	virtual ~ThreadPool();

	// Create threads and assign them to appropriate devices
	bool createThreadPool(int cpuThreads=1, int *cpuThreadsCoreMapping=NULL, int gpuThreads=0, int *gpuThreadsCoreMapping=NULL, int gpuTempDataSize=0);

	// Startup computation, so far, even if thread pool was created, the threads are awaiting for the
	// execution to be initialized. Make sure the thread poll was created before calling init execution.
	void initExecution();

	void *getGPUTempData(int tid);
	// main computation loop, where threads are kept busy computing tasks
	void processTasks(int procType, int tid);

	void finishExecWaitEnd();

	int getGPUThreads();
	int getCPUThreads();
};

#endif /* THREADPOOL_H_ */
