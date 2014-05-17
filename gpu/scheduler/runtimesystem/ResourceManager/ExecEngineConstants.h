#ifndef EXEC_ENGINE_CONSTANTS_H_
#define EXEC_ENGINE_CONSTANTS_H_

using namespace std;

class ExecEngineConstants {
public:
	//! Defining what processor should be used when invoking the functions
	static const int NUM_PROC_TYPES=2;
	static const int CPU=1;
	static const int GPU=2;

	//! Scheduling policies
	static const int FCFS_QUEUE=1;
	static const int PRIORITY_QUEUE=2;

	//! Type of task assigned to execution. Most of tasks are computing tasks,
	// while there is a special type of tasks call transaction task that is used
	// o assert that all tasks created within a given interval were finalized
	// before some action is taken
	static const int PROC_TASK=1;
	static const int TRANSACTION_TASK=2;

};

#endif
