/*
 * ExecutionEngine.cpp
 *
 *  Created on: Aug 17, 2011
 *      Author: george
 */

#include "ExecutionEngine.h"
//pthread_mutex_t ExecutionEngine::dependencyMapLock = PTHREAD_MUTEX_INITIALIZER;

ExecutionEngine::ExecutionEngine(int cpuThreads, int gpuThreads, int queueType, int gpuTempDataSize) {
	if(queueType ==ExecEngineConstants::FCFS_QUEUE){
		tasksQueue = new TasksQueueFCFS(cpuThreads, gpuThreads);
	}else{
		tasksQueue = new TasksQueuePriority(cpuThreads, gpuThreads);
	}
	threadPool = new ThreadPool(tasksQueue, this);
	threadPool->createThreadPool(cpuThreads, NULL, gpuThreads, NULL, gpuTempDataSize);
//	countTasksPending = 0;
//	this->transactionTask = NULL;
	this->trackDependencies = new TrackDependencies();
}

ExecutionEngine::~ExecutionEngine() {
	delete threadPool;
	delete tasksQueue;
	delete trackDependencies;
}

void *ExecutionEngine::getGPUTempData(int tid){
	return threadPool->getGPUTempData(tid);
}

bool ExecutionEngine::insertTask(Task *task)
{
	task->curExecEngine = this;

	// Resolve task dependencies and queue it for execution, or left the task pending waiting
	this->trackDependencies->checkDependencies(task, this->tasksQueue);

	return true;
}


Task *ExecutionEngine::getTask(int procType)
{
	return tasksQueue->getTask(procType);
}

void ExecutionEngine::startupExecution()
{
	threadPool->initExecution();
}

void ExecutionEngine::endExecution()
{
	// this protection is used just in case the user calls this function multiple times.
	// It will avoid a segmentation fault
	if(threadPool != NULL){
		tasksQueue->releaseThreads(threadPool->getGPUThreads() + threadPool->getCPUThreads());
		delete threadPool;
	}
	threadPool = NULL;
}

void ExecutionEngine::resolveDependencies(Task *task){
	// forward message to track dependencies class
	this->trackDependencies->resolveDependencies(task, this->tasksQueue);
}
//void ExecutionEngine::checkDependencies(Task *task)
//{
//	std::map<int, std::list<Task *> >::iterator dependencyMapIt;
//
//	// Lock dependency map
//	pthread_mutex_lock(&ExecutionEngine::dependencyMapLock);
//
//	for(int i = 0; i < task->getNumberDependencies(); i++){
//		// Retrieve id of the ith dependency, and check whether it is on the map: in other words, if
//		// it is still executing or waiting to execute
//		dependencyMapIt = dependencyMap.find(task->getDependency(i));
//
//		// Dependency was found in the map. Add current task to the list of task it should
//		// notify about the end of execution
//		if(dependencyMapIt != dependencyMap.end()){
//			dependencyMapIt->second.push_back(task);
//			dependencyMapIt->second.size();
//
//		}else{
//			task->incrementDepenciesSolved();
//		}
//	}
//
//	// Insert current task into the map of active tasks (those processing or pending due to dependencies)
//	list<Task *> l;
//
//	if(this->transactionTask != NULL){
//		// Okay, add taks as dependency of the transaction Task
//		l.push_back(this->transactionTask);
//		this->transactionTask->addDependency(task->getId());
//	}
//
//	dependencyMap.insert(std::pair<int, list<Task *> >(task->getId(), l));
//
//	// Check whether all dependencies were solved, and dispatches task for execution if affirmative
//	if(task->getNumberDependencies() == task->getNumberDependenciesSolved()){
//		// It always starts empty, and tasks are added as they are dispatched for execution
//		this->tasksQueue->insertTask(task);
//	}else{
//		this->incrementCountTasksPending();
//	}
//
//	// Unlock dependency map
//	pthread_mutex_unlock(&ExecutionEngine::dependencyMapLock);
//}
//
//void ExecutionEngine::resolveDependencies(Task *task)
//{
//	map<int, list<Task *> >::iterator dependencyMapIt;
//
//	// Lock dependency map
//	pthread_mutex_lock(&ExecutionEngine::dependencyMapLock);
//
//	// return data about task that has just finished
//	dependencyMapIt = dependencyMap.find(task->getId());
//	if(dependencyMapIt != dependencyMap.end()){
//
//		// Get the number of tasks depending on this one.
//		// Warning: Should not move the (dependencyMapIt->second.size()) to the loop (eliminating depsToSolve),
//		// because the size of depencyMapIt->second is modified within the loop
//		int depsToSolve = dependencyMapIt->second.size();
//
//		// for each task registered as dependency of the current task, resolve this dependency
//		for(int i = 0; i < depsToSolve;i++){
//			// takes pointer to the first dependent task
//			Task *dependentTask = dependencyMapIt->second.front();
//
//			// removes tasks from the list of dependent tasks
//			dependencyMapIt->second.pop_front();
//
//			// Increments counter regarding to the number of dependencies solved for the dependent task
//			dependentTask->incrementDepenciesSolved();
//
//			// if all dependencies of this tasks were solved, dispatches it to execution
//			if(dependentTask->getNumberDependenciesSolved() == dependentTask->getNumberDependencies()){
//				if(dependentTask->getTaskType() == ExecEngineConstants::PROC_TASK){
//					this->tasksQueue->insertTask(dependentTask);
//					this->decrementCountTasksPending();
//				}else{
//					// If all dependencies are solved, and entTransaction was called (meaning that isCallBackDesReady will return true)
//					if(dependentTask->getTaskType() == ExecEngineConstants::TRANSACTION_TASK && dependentTask->isCallBackDepsReady()){
//						std::cout << "CallBack Task"  << std::endl;
//						dependentTask->run();
//						delete dependentTask;
//					}
//				}
//
//			}
//		}
//		// Remove task that just finished from the dependency map
//		dependencyMap.erase(dependencyMapIt);
//
//	}else{
//		std::cout << "Warning: task.id="<< task->getId() << "finished execution, but its data is not available at the dependencyMap" <<std::endl;
//	}
//
//	// Unlock dependency map
//	pthread_mutex_unlock(&ExecutionEngine::dependencyMapLock);
//}

int ExecutionEngine::getCountTasksPending() const
{
    return this->trackDependencies->getCountTasksPending();
}
//
//void ExecutionEngine::incrementCountTasksPending()
//{
//	this->countTasksPending++;
//}
//
//void ExecutionEngine::decrementCountTasksPending()
//{
//	this->countTasksPending--;
//}

void ExecutionEngine::waitUntilMinQueuedTask(int numberQueuedTasks)
{
	if(numberQueuedTasks < 0) numberQueuedTasks = 0;

	// Loop waiting the number of tasks queued decrease
	while(numberQueuedTasks < tasksQueue->getSize()){
		usleep(100000);
	}

}

void ExecutionEngine::startTransaction(CallBackTaskBase *transactionTask)
{
	this->trackDependencies->startTransaction(transactionTask);
//	if(this->transactionTask != NULL){
//		std::cout << "Error: calling startTranscation before ending previous transaction (endTransaction)" <<std::endl;
//	}
//	this->transactionTask = transactionTask;
}



void ExecutionEngine::endTransaction()
{
	this->trackDependencies->endTransaction();
//	if(this->transactionTask == NULL){
//		std::cout << "Error: calling endTransaction before starting a transaction (startTranscation)" <<std::endl;
//	}else{
//
//		// Lock dependency map
//		this->trackDependencies->lock();
//
//		// if all dependencies were solved before the program executes endTransaction
//		if(this->transactionTask->getNumberDependencies() == this->transactionTask->getNumberDependenciesSolved()){
//			// All dependencies were solved, so execute callback function
//			this->transactionTask->run();
//		}
//		this->transactionTask->setCallBackDepsReady(true);
//
//		// Unlock dependency map
//		this->trackDependencies->unlock();
//	}
//	this->transactionTask = NULL;
}



