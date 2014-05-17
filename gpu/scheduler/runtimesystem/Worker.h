/*
 * Worker.h
 *
 *  Created on: Feb 15, 2012
 *      Author: george
 */

#ifndef WORKER_H_
#define WORKER_H_

#include <mpi.h>
#include <string>
#include <dlfcn.h>
#include <assert.h>
#include "MessageTag.h"
#include "PipelineComponentBase.h"
#include "CallBackComponentExecution.h"

class CallBackComponentExecution;

class Worker {

private:
	// This Workers rank
	int rank;

	// Rank assigned to the Manager process
	int manager_rank;

	// Communication group of this execution
	MPI::Intracomm comm_world;

	// Name of the library that implements the components executed by this worker
	std::string componentLibName;

	// List of components (Ids) that were completed the execution within this Worker. The
	// Resource Manager computing threads will insert items to this list using the
	// call back function that is executed when all tasks associated to a given component
	// have been successfully executed. The main Worker controlling threads will keep
	// checking whether elements were added to this list, and notify the Manger when so.
	list<int> computedComponents;

	// Used to serialize access to list of components already computed
	pthread_mutex_t computedComponentsLock;

	// Number of active component instances within this Worker.
	int activeComponentInstances;

	// Max number of active components maintained locally to this Worker.
	int maxActiveComponentInstances;

	// setter for the counter described above
    void setActiveComponentInstances(int activeComponentInstances);

	// Resource manager that executes component tasks assigned to this worker
	ExecutionEngine *resourceManager;

	// Set and Get the current resource manager
    ExecutionEngine *getResourceManager() const;
    void setResourceManager(ExecutionEngine *resourceManager);

	// Function that loads the library holding the components
    bool loadComponentsLibrary();

    // Receive information about this execution, load library, and prepare to start computation
    void configureExecutionEnvironment();

    // Receive data describing a pipeline component instantiation that should be executed
    PipelineComponentBase *receiveComponentInfoFromManager();

    // Send message to Manager notifying it about the end of components, if there are any in the list of computedComponents
    void notifyComponentsCompleted();

public:
    Worker(const MPI::Intracomm & comm_world, const int manager_rank, const int rank, const int max_active_components=1, const int CPUCores = 1, const int GPUs = 0, const int schedType = ExecEngineConstants::FCFS_QUEUE);
    virtual ~Worker();

    // Main loop that keeps receiving component instances and executing them
    void workerProcess();

    const MPI::Intracomm getCommWorld() const;

    // Simply return the values of the manager rank and this workers rank
    int getManagerRank() const;
    int getRank() const;

    // Retrieve the name of the library that implements the components used by this worker
    std::string getComponentLibName() const;

    // Set the same library name
    void setComponentLibName(std::string componentLibName);

    // Add a given component id to the output list of already computed component instances
    void addComputedComponent(int id);

    // Try to return a component id from the output list. This will return the
    // component id, or -1 whether there are not components into the output list
    int getComputedComponent();

    // Retrieve number of components in the output list
    int getComputedComponentSize();

    // Increments a counter responsible to hold the number of component instances active with the current Worker
    int incrementActiveComponentInstances();

    // Decrements the same counter previously described
    int decrementActiveComponentInstances();

    // Simple getter for the active component counter
    int getActiveComponentInstances() const;
    int getMaxActiveComponentInstances() const;
    void setMaxActiveComponentInstances(int maxActiveComponentInstances);

};


#endif /* WORKER_H_ */
