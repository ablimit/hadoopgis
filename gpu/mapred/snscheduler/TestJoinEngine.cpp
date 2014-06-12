
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

static map<string,map<int,vector<string> > > geoms;
static map<string,map<int,int> > vertexes;

float cpuSpeedUp(int nv1, int nv2, int no1, int no2){
  return 1.0 ; 
}

float gpuSpeedUp(int nv1, int nv2, int no1, int no2){
  return 5.0 ; 
}

int main(int argc, char **argv){
  if (argc <2)
  {
    std::cerr <<"Missing argument.. " << endl;
    return 0;
  }
  int loop = atoi (argv[1] ); 
  struct timeval t1, t2;
  gettimeofday(&t1, NULL);
  init_device_streams(1);
  gettimeofday(&t2, NULL);
  std::cerr<< "Time DEVICE init: " <<DIFF_TIME(t1, t2) <<" s." <<endl;
  int concurentThreadsSupported = (int)std::thread::hardware_concurrency();
  //ExecutionEngine *execEngine = new ExecutionEngine(2, 1, ExecEngineConstants::FCFS_QUEUE);
  // int nextTaskDependency;
  //std::cerr  << "Number of threads: [" << concurentThreadsSupported << "]" <<std::endl;
  // Creates first task, which does not have dependencies
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
    if (0 != tid.compare(prev_tid) && prev_tid.size()>0) 
      std::cerr << "  [" <<prev_tid << "]" ;
    // actual geometry info: did,oid,num_ver,mbb, geom
    int i = input_line[pos+1] - '1'; // array position 
    pos2=input_line.find_first_of(COMMA,pos+3); //oid = input_line.substr(pos+3,pos2-pos-3) 
    pos=input_line.find_first_of(COMMA,pos2+1); //num_ver = input_line.substr(pos2+1,pos)
    vertexes[tid][i] += std::stoi(input_line.substr(pos2+1,pos-pos2-1));
    geoms[tid][i].push_back(input_line.substr(pos2+1)); // mbb, geom
    prev_tid = tid; 
  }
  std::cerr << std::endl;

  JoinTask *jt = NULL;
    map<string,map<int,vector<string> > >::iterator it;
    map<string,map<int,int> >::iterator vertexes_iter;
  while (loop-- > 0) {
    std::cerr << "-------------------------------------------------------------------------" << std::endl;

    ExecutionEngine *execEngine = new ExecutionEngine(concurentThreadsSupported-1, 1, ExecEngineConstants::PRIORITY_QUEUE);

    // for each tile 
    for (it=geoms.begin(); it !=geoms.end(); ++it)
    {
      jt = new JoinTask(JCARDINALITY);

      for (int i = 0 ; i < JCARDINALITY; i++){
      jt->nr_vertices[i] = vertexes[it->first][i];

      jt->geom_arr[i]->assign(
          geoms[it->first][i].begin(),
          geoms[it->first][i].end());
      }
      jt->setSpeedup(ExecEngineConstants::GPU, 
                     gpuSpeedUp(jt->nr_vertices[0],
                                jt->nr_vertices[1],
                                jt->geom_arr[0]->size(), 
                                jt->geom_arr[1]->size()));
      jt->setSpeedup(ExecEngineConstants::CPU, 
                     cpuSpeedUp(jt->nr_vertices[0],
                                jt->nr_vertices[1],
                                jt->geom_arr[0]->size(), 
                                jt->geom_arr[1]->size()));
      // Dispatches current tasks for execution
      //std::cerr << ts->getId() << "-------" << prev_tid<< std::endl;
      execEngine->insertTask(jt);
    }

    // Computing threads startup consuming tasks
    execEngine->startupExecution();

    // No more task will be assigned for execution. Waits
    // until all currently assigned have finished.
    execEngine->endExecution();

    delete execEngine;
  }
  fini_device_streams(1);
  return 0;
}


