/*
 * SysEnv.cpp
 *
 *  Created on: Feb 15, 2012
 *      Author: george
 */

#include "SysEnv.h"
#include <iostream>
#include <stdio.h>
#include <vector>
#include <string>

#include <mpi.h>

#include "Util.h"
#include "Manager.h"
#include "Worker.h"
#include <stdlib.h>

SysEnv::SysEnv() {
	this->manager = NULL;

}

SysEnv::~SysEnv() {
	if(this->manager != NULL){
		delete this->manager;
	}
}

// initialize MPI
MPI::Intracomm init_mpi(int argc, char **argv, int &size, int &rank, std::string &hostname) {
    MPI::Init(argc, argv);

    char *temp = new char[256];
    gethostname(temp, 255);
    hostname.assign(temp);
    delete [] temp;

    size = MPI::COMM_WORLD.Get_size();
    rank = MPI::COMM_WORLD.Get_rank();

    return MPI::COMM_WORLD;
}

Manager *SysEnv::getManager() const
{
    return manager;
}

void SysEnv::setManager(Manager *manager)
{
    this->manager = manager;
}

int SysEnv::startupSystem(int argc, char **argv, std::string componentsLibName){
	// set up mpi
	int rank, size, worker_size, manager_rank;
	std::string hostname;

	MPI::Intracomm comm_world = init_mpi(argc, argv, size, rank, hostname);

	if (size == 1) {
		printf("ERROR:  this program can only be run with 2 or more MPI nodes.  The head node does not process data\n");
		exit(1);
		return -4;
	}

	// initialize the worker comm object
	worker_size = size - 1;
	manager_rank = size - 1;


	uint64_t t1 = 0, t2 = 0;
	t1 = Util::ClockGetTime();

	// decide based on rank of worker which way to process
	if (rank == manager_rank) {
		// Create the manager process information
		this->setManager(new Manager(comm_world, manager_rank, worker_size));

		// Check whether all Worker have successfully initialized their execution
		this->getManager()->checkConfiguration();

		// Return the Manager control flow the the user code, that
		// will presumably instantiate the pipeline for execution
		return 0;

	} else {
		// Create one worker object for each Worker process
		Worker* localWorker = new Worker(comm_world, manager_rank, rank, 5, 4 );

		// Initialize the name of the library containing the Pipeline components
		localWorker->setComponentLibName(componentsLibName);

		// This is the main computation loop, where the Worker
		// will keep asking for tasks and executing them.
		localWorker->workerProcess();

		// Delete Worker structures
		delete localWorker;
	}

	// Shake hands and finalize MPI
	comm_world.Barrier();
	MPI::Finalize();
	exit(0);

}

int SysEnv::startupExecution(){
	uint64_t t1 = 0, t2 = 0;
	printf("Manager StartupExecution");
	t1 = Util::ClockGetTime();

	this->getManager()->manager_process();

	t2 = Util::ClockGetTime();

	printf("MANAGER %d : FINISHED in %llu us\n", this->getManager()->getManagerRank(), t2 - t1);
}

int SysEnv::finalizeSystem()
{
	return this->getManager()->finalizeExecution();
}

int SysEnv::executeComponent(PipelineComponentBase* compInstance) {
	this->getManager()->insertComponentInstance(compInstance);
	return 0;
}



