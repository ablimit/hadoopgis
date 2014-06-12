
#include <stdio.h>
#include <thread>
#include <sys/time.h>
#include "JoinTask.h"
#include "ExecutionEngine.h"

#define NUM_TASKS	6
#define JCARDINALITY 2 
static const string TAB = "\t";
static const string COMMA = ",";
static const string SPACE = " ";

float cpuSpeedUp(int nv1, int nv2, int no1, int no2){
  return 1.0 ; 
}

float gpuSpeedUp(int nv1, int nv2, int no1, int no2){
  return 4.993 - 9.722 * 0.00001 *no1 + 2.455 * 0.001 * no2 - 2.604 * 0.000001*nv1 - 2.434 * 0.00001* nv2 ;
}

int main(int argc, char **argv){

  struct timeval t1, t2;
  gettimeofday(&t1, NULL);
  init_device_streams(1);
  gettimeofday(&t2, NULL);
  std::cerr<< "Time DEVICE init: " <<DIFF_TIME(t1, t2) <<" s." <<endl;
  int concurentThreadsSupported = (int)std::thread::hardware_concurrency();
  ExecutionEngine *execEngine = new ExecutionEngine(concurentThreadsSupported, 1, ExecEngineConstants::PRIORITY_QUEUE);
  //ExecutionEngine *execEngine = new ExecutionEngine(2, 1, ExecEngineConstants::FCFS_QUEUE);
  // int nextTaskDependency;
  //std::cerr  << "Number of threads: [" << concurentThreadsSupported << "]" <<std::endl;
  // Creates first task, which does not have dependencies
  JoinTask *ts = new JoinTask(JCARDINALITY);
  size_t pos, pos2;
  string input_line;
  string tid ;
  string prev_tid = "";

  std::cerr << "I/O" ;
  while(cin && getline(cin, input_line) && !cin.eof()) {
    pos=input_line.find_first_of(TAB,0);
    if (pos == string::npos){
      cerr << "no TAB in the input! We are toasted." << endl;
      return 1; // failure
    }

    tid= input_line.substr(0,pos); // tile id

    // finished reading in a tile data, so perform cross matching
    if (0 != tid.compare(prev_tid) && prev_tid.size()>0) 
    {
      ts->setSpeedup(ExecEngineConstants::GPU, 
                     gpuSpeedUp(ts->nr_vertices[0],
                                ts->nr_vertices[1],
                                ts->geom_arr[0]->size(), 
                                ts->geom_arr[1]->size()));
      ts->setSpeedup(ExecEngineConstants::CPU, 
                     cpuSpeedUp(ts->nr_vertices[0],
                                ts->nr_vertices[1],
                                ts->geom_arr[0]->size(), 
                                ts->geom_arr[1]->size()));
      // Dispatches current tasks for execution
      //std::cerr << ts->getId() << "-------" << prev_tid<< std::endl;
      execEngine->insertTask(ts);
      ts = new JoinTask(JCARDINALITY);
      std::cerr << "  [" <<prev_tid << "]" ;
    }
    // actual geometry info: did,oid,num_ver,mbb, geom
    int i = input_line[pos+1] - '1'; // array position 
    pos2=input_line.find_first_of(COMMA,pos+3); //oid = input_line.substr(pos+3,pos2-pos-3) 
    pos=input_line.find_first_of(COMMA,pos2+1); //num_ver = input_line.substr(pos2+1,pos)
    ts->nr_vertices[i] += std::stoi(input_line.substr(pos2+1,pos-pos2-1));
    ts->geom_arr[i]->push_back(input_line.substr(pos2+1)); // mbb, geom
    prev_tid = tid; 
  }
  ts->setSpeedup(ExecEngineConstants::CPU, 
                     gpuSpeedUp(ts->nr_vertices[0],
                                ts->nr_vertices[1],
                                ts->geom_arr[0]->size(), 
                                ts->geom_arr[1]->size()));
  ts->setSpeedup(ExecEngineConstants::GPU, 
                 gpuSpeedUp(ts->nr_vertices[0],
                            ts->nr_vertices[1],
                            ts->geom_arr[0]->size(), 
                            ts->geom_arr[1]->size()));
  // Dispatches current tasks for execution
  execEngine->insertTask(ts);
  std::cerr << "  ["<<tid << "]" << std::endl;

  // Computing threads startup consuming tasks
  execEngine->startupExecution();

  // No more task will be assigned for execution. Waits
  // until all currently assigned have finished.
  execEngine->endExecution();

  fini_device_streams(1);
  delete execEngine;

  return 0;
}


