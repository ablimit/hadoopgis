#include "JoinTask.h"

JoinTask::JoinTask(int n): N(n) {
  
  geom_arr = new vector<string>*[N];
  
  for (int i =0 ; i < N ; i++)
  {
    geom_arr[i] = new vector<string>();
    nr_vertices.push_back(0);
  }
}

JoinTask::~JoinTask() {
  std::cerr << "~JoinTask" << std::endl;
  
  for (int i =0 ; i < N ; i++)
    delete geom_arr[i];
  
  delete [] geom_arr ;
  
  nr_vertices.clear();
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
  //this->printDependencies();
  return true;
}

