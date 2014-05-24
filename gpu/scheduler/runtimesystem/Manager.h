/*
 * Manager.h
 *
 *  Created on: Feb 15, 2012
 *      Author: george
 */

#ifndef MANAGER_H_
#define MANAGER_H_

#include <mpi.h>
#include <vector>
#include <assert.h>
#include <stdlib.h>
#include <stdint.h>

#include "MessageTag.h"
#include "PipelineComponentBase.h"

class PipelineComponentBase;

class Manager {
private:
	// Number of Worker processes
	int worker_size;

	// Ranking of this manager within the communication group
	int manager_rank;

	// Components with dependencies solved, meaning ready to execute
	TasksQueue* componentsToExecute;

	// Mechanism that tracks dependencies among component instances
	TrackDependencies* componentDependencies;

	// Components that were sent for execution and did not finish yet
	map<int, PipelineComponentBase *> activeComponents;

	int insertActiveComponent(PipelineComponentBase* pc);
	PipelineComponentBase* retrieveActiveComponent(int id);
	int getActiveComponentsSize();

	// MPI structure defining the communication group information
	MPI::Intracomm comm_world;

	void sendComponentInfoToWorker(int worker_id, PipelineComponentBase *pc);

	int resolveDependencies(PipelineComponentBase *pc);

	friend class SysEnv;
	friend class PipelineComponentBase;

public:
	Manager(const MPI::Intracomm& comm_world, const int manager_rank, const int worker_size, const int queueType=ExecEngineConstants::FCFS_QUEUE);
	virtual ~Manager();

	void manager_process();
	int finalizeExecution();
	void checkConfiguration();

    MPI::Intracomm getCommWorld() const;
    int getManagerRank() const;
    int getWorkerSize() const;
    void setWorkerSize(int worker_size);

    int insertComponentInstance(PipelineComponentBase * compInstance);


};

#endif /* MANAGER_H_ */
