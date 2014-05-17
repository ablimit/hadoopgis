/*
 * Manager.cpp
 *
 *  Created on: Feb 15, 2012
 *      Author: george
 */

#include "Manager.h"

Manager::Manager(const MPI::Intracomm& comm_world, const int manager_rank, const int worker_size, const int queueType) {
	this->comm_world =comm_world;
	this->manager_rank = manager_rank;
	this->worker_size = worker_size;
	if(queueType ==ExecEngineConstants::FCFS_QUEUE){
		componentsToExecute = new TasksQueueFCFS(1, 0);
	}else{
		componentsToExecute = new TasksQueuePriority(1, 0);
	}
	this->componentDependencies = new TrackDependencies();

}

Manager::~Manager() {
	if(componentsToExecute != NULL){
		delete componentsToExecute;
	}
}

MPI::Intracomm Manager::getCommWorld() const
{
    return comm_world;
}

int Manager::getManagerRank() const
{
    return manager_rank;
}

int Manager::getWorkerSize() const
{
    return worker_size;
}

int Manager::finalizeExecution()
{
	MPI::Status status;
	int worker_id;
	char ready;

	/* tell everyone to quit */
	int active_workers = worker_size;
	while (active_workers > 0) {
		usleep(1000);

		if (comm_world.Iprobe(MPI_ANY_SOURCE, MessageTag::TAG_CONTROL, status)) {

			// where is it coming from
			worker_id=status.Get_source();
			comm_world.Recv(&ready, 1, MPI::CHAR, worker_id, MessageTag::TAG_CONTROL);

			if (worker_id == manager_rank) continue;

			if(ready == MessageTag::WORKER_READY) {
				comm_world.Send(&MessageTag::MANAGER_FINISHED, 1, MPI::CHAR, worker_id, MessageTag::TAG_CONTROL);
				//				printf("manager signal finished\n");
				--active_workers;
			}
		}
	}
	this->getCommWorld().Barrier();
	MPI::Finalize();
	return 0;
}

void Manager::checkConfiguration()
{

	MPI::Status status;
	int worker_id;
	char ready;
	bool correctInitialization = true;

	int active_workers = worker_size;

	// Listening from each worker whether it was initialized correctly
	while (active_workers > 0) {
		usleep(1000);

		if (comm_world.Iprobe(MPI_ANY_SOURCE, MessageTag::TAG_CONTROL, status)) {

			// where is it coming from
			worker_id=status.Get_source();
			bool curWorkerStatus;
			comm_world.Recv(&curWorkerStatus, 1, MPI::BOOL, worker_id, MessageTag::TAG_CONTROL);
			if(curWorkerStatus == false){
				correctInitialization = false;
			}
			active_workers--;
		}
	}

	// Tell each worker whether to continue or quit the execution
	comm_world.Bcast(&correctInitialization, 1 , MPI::BOOL, this->getManagerRank());

	if(correctInitialization==false){
		std::cout << "Quitting. Workers initialization failed. Possible due to errors loading the components library."<<std::endl;
		MPI::Finalize();
		exit(1);
	}
}

void Manager::sendComponentInfoToWorker(int worker_id, PipelineComponentBase *pc)
{
	std::cout << "Manager: Sending component id="<< pc->getId() <<std::endl;
	int comp_serialization_size = pc->size();
	char *buff = new char[comp_serialization_size];
	int used_serialization_size = pc->serialize(buff);
	assert(comp_serialization_size == used_serialization_size);

	comm_world.Send(buff, comp_serialization_size, MPI::CHAR, worker_id, MessageTag::TAG_METADATA);

	delete[] buff;
}

void Manager::setWorkerSize(int worker_size)
{
    this->worker_size = worker_size;
}

void Manager::manager_process()
{

	uint64_t t1, t0;

	std::cout<< "Manager ready. Rank = %d"<<std::endl;
	comm_world.Barrier();

	// now start the loop to listen for messages
	int curr = 0;
	int total = 10;
	MPI::Status status;
	int worker_id;
	char msg_type;
	int inputlen = 15;

	//TODO: testing only
	int tasksToFinishTasks = componentsToExecute->getSize();
	std::cout << __FILE__ << ":" << __LINE__ << ". TasksToExecute="<<tasksToFinishTasks<<std::endl;

	// Process all components instantiated for execution
	while (componentsToExecute->getSize() != 0 || this->componentDependencies->getCountTasksPending() != 0 || this->getActiveComponentsSize()) {

		if (comm_world.Iprobe(MPI_ANY_SOURCE, MessageTag::TAG_CONTROL, status)) {

			// Where is the message coming from
			worker_id=status.Get_source();

			// Check the size of the input message
			int input_message_size = status.Get_count(MPI::CHAR);

			assert(input_message_size > 0);

			char *msg = new char[input_message_size];

			// Read the
			comm_world.Recv(msg, input_message_size, MPI::CHAR, worker_id, MessageTag::TAG_CONTROL);
			//			printf("manager received request from worker %d\n",worker_id);
			msg_type = msg[0];

			switch(msg_type){
				case MessageTag::WORKER_READY:
				{
					if(this->componentsToExecute->getSize() > 0){
						// select next component instantiation should be dispatched for execution
						PipelineComponentBase *compToExecute = (PipelineComponentBase*)componentsToExecute->getTask();

						// tell worker that manager is ready
						comm_world.Send(&MessageTag::MANAGER_READY, 1, MPI::CHAR, worker_id, MessageTag::TAG_CONTROL);
						std::cout << "Manager: before sending, size: "<< this->componentsToExecute->getSize() << std::endl;
						this->sendComponentInfoToWorker(worker_id, compToExecute);

						this->insertActiveComponent(compToExecute);

					}else{
						// tell worker that manager queue is empty. Nothing else to do at this moment. Should ask again.
						comm_world.Send(&MessageTag::MANAGER_WORK_QUEUE_EMPTY, 1, MPI::CHAR, worker_id, MessageTag::TAG_CONTROL);
					}
					break;
				}
				case MessageTag::WORKER_TASKS_COMPLETED:
				{
					// Pointer to the message area where the information about the tasks is stored
					int *tasks_data = (int*)(msg+sizeof(char));
					int number_components_completed = tasks_data[0];
					std::cout << "Manager: #CompCompleted = "<< number_components_completed <<std::endl;

					// Iterate over the component instances that were completed
					for(int i = 0; i < number_components_completed; i++){
						std::cout << "	id="<< tasks_data[i+1] << std::endl;

						// Retrieve component that just finished the execution, and delete it to resolve dependencies
						PipelineComponentBase * compPtrAux = this->retrieveActiveComponent(tasks_data[i+1]);

						// Assert that component was correctly found in the map of active components
						if(compPtrAux != NULL){
							delete compPtrAux;
						}else{
							std::cout << __FILE__ << ":"<< __LINE__ <<". Error: Component Id="<< tasks_data[i+1] << " not found!" <<std::endl;
						}
					}

					break;
				}
				default:
					std::cout << "Unknown message type="<< (int)msg_type <<std::endl;
					break;
			}

			delete[] msg;
		}else{
			usleep(1000);
		}

	}

}

int Manager::insertComponentInstance(PipelineComponentBase* compInstance) {

//	compInstance->curExecEngine = NULL;
	compInstance->managerContext = this;

	// Resolve component dependencies and queue it for execution, or left the component pending waiting
	this->componentDependencies->checkDependencies(compInstance, this->componentsToExecute);

	return 0;
}

int Manager::insertActiveComponent(PipelineComponentBase* pc) {
	int retValue = 0;
	assert(pc != NULL);

	this->activeComponents.insert(std::pair<int, PipelineComponentBase * >(pc->getId(), pc));
	return retValue;
}

PipelineComponentBase* Manager::retrieveActiveComponent(int id) {
	PipelineComponentBase* compRet = NULL;
	map<int, PipelineComponentBase*>::iterator activeCompIt;

	// Try to find component id that just finished
	activeCompIt = this->activeComponents.find(id);

	// If id is found, return component and removed it from the map.
	if(activeCompIt != this->activeComponents.end()){

		// Just assign to the return point the value of the component found
		compRet = activeCompIt->second;

		// Remove component instance from the map
		this->activeComponents.erase(activeCompIt);
	}
	return compRet;
}

int Manager::getActiveComponentsSize() {
	return this->activeComponents.size();
}

int Manager::resolveDependencies(PipelineComponentBase* pc) {
	this->componentDependencies->resolveDependencies(pc, this->componentsToExecute);
	return 0;
}






