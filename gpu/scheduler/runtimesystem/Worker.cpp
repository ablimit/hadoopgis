/*
 * Worker.cpp
 *
 *  Created on: Feb 15, 2012
 *      Author: george
 */

#include "Worker.h"

Worker::Worker(const MPI::Intracomm& comm_world, const int manager_rank, const int rank, const int max_active_components, const int CPUCores, const int GPUs, const int schedType) {
	this->manager_rank = manager_rank;
	this->rank = rank;
	this->comm_world = comm_world;
	this->setMaxActiveComponentInstances(max_active_components);
	this->setActiveComponentInstances(0);

	// Create a local Resource Manager
	this->setResourceManager(new ExecutionEngine(CPUCores, GPUs, schedType));

	// Computing threads startup consuming tasks
	this->getResourceManager()->startupExecution();

	pthread_mutex_init(&this->computedComponentsLock, NULL);
}


Worker::~Worker() {

	// No more task will be assigned for execution. Waits until all currently assigned have finished.
	this->getResourceManager()->endExecution();

	// Delete the Resource Manager
	delete this->getResourceManager();

	pthread_mutex_destroy(&this->computedComponentsLock);
}

const MPI::Intracomm Worker::getCommWorld() const
{
    return comm_world;
}

int Worker::getManagerRank() const
{
    return manager_rank;
}



std::string Worker::getComponentLibName() const
{
    return componentLibName;
}

void Worker::setComponentLibName(std::string componentLibName)
{
    this->componentLibName = componentLibName;
}

bool Worker::loadComponentsLibrary()
{
	bool retValue = true;
	char *error = NULL;
	std::cout<< "Load ComponentsLibrary(). Libname="<<this->getComponentLibName()<<std::endl;

	// try load lib in local directory, so we need 2 strigns,
	// the first with ./, em the second whithout
	std::string libNameLocal;
	libNameLocal.append("./");
	libNameLocal.append(this->getComponentLibName());

	// get the library handler
	void* componentLibHandler = NULL;
	if (((componentLibHandler = dlopen(this->getComponentLibName().c_str(), RTLD_NOW)) == NULL) &&
		((componentLibHandler = dlopen(libNameLocal.c_str(), RTLD_NOW)) == NULL )) {
			fprintf(stderr, "Could not Components %s library, %s\n", this->getComponentLibName().c_str(), dlerror());
			dlclose(componentLibHandler);
			std::cout << "Could not load library components:"<<this->getComponentLibName()<<std::endl;
			retValue =false;
	}else{
		std::cout << "Library Components successfully load" <<std::endl;
	}

//	componetFactory_t* compFactory = PipelineComponentBase::ComponentFactory::getComponentFactory("CompPrint");
//	PipelineComponentBase *pipelineComponent = compFactory();
//	pipelineComponent->run();

	return retValue;
}

void Worker::configureExecutionEnvironment(){

	// Load component library
	bool successConf = this->loadComponentsLibrary();

	// Sent configuration status to the Manager: so far, it only means
	// that the components library was correctly initialized
	comm_world.Send(&successConf, 1, MPI::BOOL, this->getManagerRank(), MessageTag::TAG_CONTROL);

	// Receive global initialization result from the Manager
	comm_world.Bcast(&successConf, 1 , MPI::BOOL, this->getManagerRank());

	// Finalize the execution if the initialization failed.
	if(successConf==false){
		std::cout << "Quitting. Initialization failed, Worker rank:"<< this->getRank() <<std::endl;
		MPI::Finalize();
		exit(1);
	}
}

PipelineComponentBase *Worker::receiveComponentInfoFromManager()
{
	PipelineComponentBase *pc = NULL;
	MPI::Status status;
	int message_size;

	// Probe for incoming message from Manager
	this->comm_world.Probe(this->getManagerRank(), (int)MessageTag::TAG_METADATA, status);

	// Check the size of the input message
	message_size = status.Get_count(MPI::CHAR);

	char *msg = new char[message_size];

//	std::cout << "Msg size="<<message_size<<std::endl;

	// get data from manager
	this->comm_world.Recv(msg, message_size, MPI::CHAR, this->getManagerRank(), MessageTag::TAG_METADATA);

	// Unpack the name of the component to instantiate it and deserialize message
	int comp_name_size = ((int*)msg)[1];
	char *comp_name = new char[comp_name_size+1];
	comp_name[comp_name_size] = '\0';
	memcpy(comp_name, msg+(2*sizeof(int)), comp_name_size*sizeof(char));
//	std::cout << "CompName="<< comp_name <<std::endl;

	pc = PipelineComponentBase::ComponentFactory::getCompoentFromName(comp_name);
	if(pc != NULL){
		pc->deserialize(msg);
	}else{
		std::cout << "Error reading component name="<< comp_name << std::endl;

	}
	delete[] msg;
	delete[] comp_name;
	return pc;
}

ExecutionEngine *Worker::getResourceManager() const
{
    return this->resourceManager;
}

void Worker::setResourceManager(ExecutionEngine *resourceManager)
{
    this->resourceManager = resourceManager;
}

int Worker::getRank() const
{
    return rank;
}


void Worker::workerProcess()
{
	// Load Components implemented by the current application, and
	// check if all workers were correctly initialized.
	this->configureExecutionEnvironment();

	std::cout << "Worker: " << this->getRank() << ", before ready Barrier" << std::endl;

	// Wait until the Manger startups the execution
	this->comm_world.Barrier();

	std::cout << "Worker: " << this->getRank() << " ready" << std::endl;

	// Flag that control the execution loop, and is updated from messages sent by the Manager
	char flag = MessageTag::MANAGER_READY;
	while (flag != MessageTag::MANAGER_FINISHED && flag != MessageTag::MANAGER_ERROR) {

		// tell the manager - ready
		this->comm_world.Send(&MessageTag::WORKER_READY, 1, MPI::CHAR, this->getManagerRank(), MessageTag::TAG_CONTROL);

		// get the manager status
		this->comm_world.Recv(&flag, 1, MPI::CHAR, this->getManagerRank(), MessageTag::TAG_CONTROL);

		switch(flag){
			case MessageTag::MANAGER_READY:
			{

				PipelineComponentBase *pc = this->receiveComponentInfoFromManager();

				if(pc != NULL){
					// One more component instance was received and is being dispatched for execution
					this->incrementActiveComponentInstances();

					// Associated local resource manager to the received component
					pc->setResourceManager(this->getResourceManager());

					// Create a Transaction tasks relates subtasks created by this component to itself. When all
					// subtasks have finished the callback is executed, destroyed and the component is destroyed
					// after all related subtasks have finished

					CallBackComponentExecution *callBackTask = new CallBackComponentExecution(pc, this);

					// Start transaction. All tasks created within the execution engine will be associated to this one
					this->getResourceManager()->startTransaction(callBackTask);

					// Execute component function that instantiated tasks within the execution engine
					pc->run();

					// Stop transaction: defines the end of the transaction associated to the current component
					this->getResourceManager()->endTransaction();

					// Once the component was executed to created the subtasks it is no longer necessary anymore
					//delete pc;
				}else{
					std::cout << "Error: Failed to load PipelineComponent!"<<std::endl;
				}
				break;
			}
			case MessageTag::MANAGER_FINISHED:
			{
				std::cout << "Manager finished execution. Worker id="<< this->getRank() <<std::endl;
				break;
			}
			case MessageTag::MANAGER_WORK_QUEUE_EMPTY:
			{
				// Wait until ask for some work again
				usleep(10000);

				break;
			}
			default:
			{
				std::cout << "Manager:"<< __FILE__ << ":"<< __LINE__ << ". Unknown message type:"<< (int)flag<<std::endl;
				break;
			}
		}

		this->notifyComponentsCompleted();
		// Okay. Reached the limit of component instances that could be concurrently active
		// at this Worker. Wait until at least one instance is finished
		while( this->getMaxActiveComponentInstances() == this->getActiveComponentInstances()){

			// Wait some time before trying again. Avoiding busy wait here.
			usleep(10000);

			// check whether a component has finished, and notify the Manager about that
			this->notifyComponentsCompleted();

		}
	}
	// Assert that all tasks queued for executed are completed before leave
	this->getResourceManager()->endExecution();



	std::cout<< "ComputedComponents = "<< this->getComputedComponentSize() <<std::endl;

}

void Worker::addComputedComponent(int id)
{
	// Get list lock before accessing it
	pthread_mutex_lock(&this->computedComponentsLock);

	this->computedComponents.push_back(id);

	// Release list lock
	pthread_mutex_unlock(&this->computedComponentsLock);
}



int Worker::getComputedComponentSize()
{
	int computedComponentSize = 0 ;

	// Lock access to computed components list
	pthread_mutex_lock(&this->computedComponentsLock);

	computedComponentSize = this->computedComponents.size();

	// Release list lock
	pthread_mutex_unlock(&this->computedComponentsLock);
	return computedComponentSize;
}

void Worker::notifyComponentsCompleted()
{
	if(this->getComputedComponentSize() > 0){
		// Lock access to computed components list
		pthread_mutex_lock(&this->computedComponentsLock);

		// Number of components I need to pack
		int number_components = this->computedComponents.size();

		// Message format
		// |Mesg Type (char) | Number of Components to Pack (int) | the proper component ids (int * Number of Components to Pack)
		int message_size = sizeof(char) + sizeof(int) + number_components * sizeof(int);

		// Allocate Message
		char *msg = new char[message_size];

		// Pack type of message
		msg[0] = MessageTag::WORKER_TASKS_COMPLETED;

		// Pointer to (int) part of the message
		int *int_data_ptr = (int*)(msg + sizeof(char));

		// Pack number of components finished
		int_data_ptr[0] = number_components;

		// Pack ids of all components
		for(int i = 0; i < number_components; i++){
			// Pack ith component id
			int_data_ptr[i+1] = this->computedComponents.front();

			// Removed ith component from the output list
			this->computedComponents.pop_front();
			this->decrementActiveComponentInstances();

		}

		// Release list lock
		pthread_mutex_unlock(&this->computedComponentsLock);

		// Now we have a message (msg) that is ready to sent to the Master
		this->comm_world.Send(msg, message_size, MPI::CHAR, this->getManagerRank(), MessageTag::TAG_CONTROL);

		// Remove message
		delete[] msg;


	}
}

int Worker::getMaxActiveComponentInstances() const
{
    return maxActiveComponentInstances;
}

void Worker::setMaxActiveComponentInstances(int maxActiveComponentInstances)
{
	assert(maxActiveComponentInstances > 0);
    this->maxActiveComponentInstances = maxActiveComponentInstances;
}

int Worker::getActiveComponentInstances() const
{
    return activeComponentInstances;
}

void Worker::setActiveComponentInstances(int activeComponentInstances)
{
    this->activeComponentInstances = activeComponentInstances;
}

int Worker::getComputedComponent()
{
	int componentId = -1;;
	// Get list lock before accessing it
	pthread_mutex_lock(&this->computedComponentsLock);

	if(this->computedComponents.size() > 0){
		componentId = this->computedComponents.front();
		this->computedComponents.pop_front();
	}

	// Release list lock
	pthread_mutex_unlock(&this->computedComponentsLock);
	return componentId;
}

int Worker::incrementActiveComponentInstances()
{
	this->activeComponentInstances++;
	return this->activeComponentInstances;
}



int Worker::decrementActiveComponentInstances()
{
	this->activeComponentInstances--;
	return this->activeComponentInstances;
}







