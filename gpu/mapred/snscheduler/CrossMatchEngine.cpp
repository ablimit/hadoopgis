
#include <stdio.h>
#include <thread>
#include <sys/time.h>
#include "TaskId.h"
#include "ExecutionEngine.h"

#define NUM_TASKS	6

int main(int argc, char **argv){
	
	ExecutionEngine *execEngine = new ExecutionEngine(2, 1, ExecEngineConstants::PRIORITY_QUEUE);

	int nextTaskDependency;
  unsigned concurentThreadsSupported = std::thread::hardware_concurrency();
  std::cerr  << "Number of threads: [" << concurentThreadsSupported << "]" <<std::endl;
	// Creates first task, which does not have dependencies
	TaskId *ts = new TaskId();
	ts->setSpeedup(ExecEngineConstants::GPU, 1.0);
	// Dispatches current tasks for execution
	execEngine->insertTask(ts);

	// Gets Id of the current task to set as dependency of the following
	nextTaskDependency = ts->getId();

	// Create a second task without dependencies
	TaskId *ts1 = new TaskId();
	int seconTaskId = ts1->getId();
  ts1->addDependency(nextTaskDependency);
	execEngine->insertTask(ts1);


	// Computing threads startup consuming tasks
	execEngine->startupExecution();

	// No more task will be assigned for execution. Waits
	// until all currently assigned have finished.
	execEngine->endExecution();
	
	delete execEngine;
	return 0;
}


