#include "JoinTask.h"

JoinTask::JoinTask() {
}

JoinTask::~JoinTask() {
  std::cerr << "~JoinTask" << std::endl;
}

bool JoinTask::run(int procType, int tid)
{
  if (procType == ExecEngineConstants::CPU) {
    sleep(2);
    std::cerr << "executing on the CPU engine. " << std::endl;

  }
  else if( procType == ExecEngineConstants::GPU ) {

    sleep(1);
    std::cerr << "executing on the GPU engine. " << std::endl;

    //sleep(5/this->getSpeedup(ExecEngineConstants::GPU));
  }
  else 
    std::cout << "No idea how to handle. " << std::endl;
  //	std::cout << "Task.id = "<< this->getId() << std::endl;
  this->printDependencies();
  return true;
}

