/*
 * SysEnv.h
 *
 *  Created on: Feb 15, 2012
 *      Author: george
 */

#ifndef SYSENV_H_
#define SYSENV_H_

#include <string>
#include "Types.hpp"
#include "Manager.h"
#include "Argument.h"


class SysEnv {
private:
	Manager* manager;

	Manager *getManager() const;
    void setManager(Manager *manager);

public:
	SysEnv();
	virtual ~SysEnv();

	int startupSystem(int argc, char **argv, std::string componentsLibName);

	int executeComponent(PipelineComponentBase *compInstance);

	int startupExecution();
    int finalizeSystem();

};

#endif /* SYSENV_H_ */
