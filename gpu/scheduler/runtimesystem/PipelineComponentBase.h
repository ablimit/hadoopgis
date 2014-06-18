/*
 * PipelineComponentBase.h
 *
 *  Created on: Feb 16, 2012
 *      Author: george
 */

#ifndef PIPELINECOMPONENTBASE_H_
#define PIPELINECOMPONENTBASE_H_

#include <vector>
#include <string>
#include <map>

#include "Argument.h"
#include "Task.h"
#include "Manager.h"


class PipelineComponentBase;
class Manager;

// Define factory function type that creates objects of type PipelineComponentBase and its subclasses
typedef PipelineComponentBase* (componetFactory_t)();

class PipelineComponentBase: public Task {
private:
	// Contain pointers to all arguments associated to this pipeline component
	std::vector<ArgumentBase*> arguments;

	// Holds the string name of the component, which should be the same used to register with the ComponentFactory
	std::string component_name;

	Manager *managerContext;

	// Resource manager used to execute this pipeline. This pointer is initialized at
	// the node level, when a Worker instantiates the pipeline component
	ExecutionEngine *resourceManager;

	// Simply set, and get resourceManager value
    void setResourceManager(ExecutionEngine *resourceManager);
    ExecutionEngine *getResourceManager() const;

	// Unique identifier of the class instance.
	int id;

	// Auxiliary class variable used to assign an unique id to each class object instance.
	static int instancesIdCounter;

	friend class Worker;
	friend class Manager;

public:
	PipelineComponentBase();
	virtual ~PipelineComponentBase();

	// This is the function implemented by the user in the subclasses of this one, and this
	// is the function executed by the runtime system which should contain the computation
	// associated to this pipeline component. Presumably, exposed a pipeline of tasks that
	// are further assigned to the Resource Manager.
	virtual int run(){return 1;};

	// Add an argument to the end of the list
	void addArgument(ArgumentBase *arg);

	// Retrieve "index"th argument, if it exists, otherwise NULL is returned
	ArgumentBase *getArgument(int index);

	// Get current number of arguments associated to this component.
	int getArgumentsSize();

	// Return name of the component
    std::string getComponentName() const;

    // Yep, set the name of the component
    void setComponentName(std::string component_name);

    // Serialization size: number of bytes need to store components
    int size();

    // Write component data to a buffer
    int serialize(char *buff);

    // Initialize component data from a buffer generated by serialize function
    int deserialize(char *buff);

    // Return Id to this component instance
	int getId() const;

	// Set component instance id
	void setId(int id);

	// Dispatch task for execution
	void executeTask(Task *task);

	Manager* getManagerContext() const {
		return managerContext;
	}

	void setManagerContext(Manager* managerContext) {
		this->managerContext = managerContext;
	}

	// Factory class is used to build "reflection", and instantiate objects of
	// PipelineComponentBase subclasses that register with it
    class ComponentFactory{
    private:
		// This maps name of component types to the function that creates instances of those components
    	static std::map<std::string,componetFactory_t*> factoryMap;

    public:
    	// Used to register the component factory function with this factory class
    	static bool componentRegister(std::string name, componetFactory_t *compFactory);

    	// Retrieve pointer to function that creates components registered with name="name"
    	static componetFactory_t *getComponentFactory(std::string name);

    	// Retrieve instance of component registered as "name"
    	static PipelineComponentBase *getCompoentFromName(std::string name);
        ExecutionEngine *getResourceManager() const;

    };
};

//#define componentFactoryFunction( c1 )\
//	PipelineComponentBase* componentFactory() {\
//		return new c1();\
//	}\

#endif /* PIPELINECOMPONENTBASE_H_ */