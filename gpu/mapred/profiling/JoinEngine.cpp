
#include <stdio.h>
#include <thread>
#include <sys/time.h>
#include <iostream>
#include <fstream>
#include <sstream>

#include "JoinTask.h"
#include "ExecutionEngine.h"

#define NUM_TASKS	6
#define JCARDINALITY 2 
static const string TAB = "\t";
static const string COMMA = ",";
static const string SPACE = " ";

int main(int argc, char **argv){
  bool gpuEngine = false;
  if (argc <2 )
  {
    cerr << "Error: missing tile input file."<<endl;
    exit(1);
  }
  if (argc>2)
    gpuEngine= true;

  struct timeval t1, t2;
  gettimeofday(&t1, NULL);
  init_device_streams(1);
  JoinTask *ts = new JoinTask(JCARDINALITY);
  size_t pos, pos2;
  string input_line;
  string tid ;
  string prev_tid = "";

  //while(cin && getline(cin, input_line) && !cin.eof()) {
  std::ifstream infile(argv[1]);
  while(getline(infile, input_line)) {
    pos=input_line.find_first_of(TAB,0);
    if (pos == string::npos){
      cerr << "no TAB in the input! We are toasted." << endl;
      return 1; // failure
    }

    tid= input_line.substr(0,pos); // tile id

    // finished reading in a tile data, so perform cross matching
    if (0 != tid.compare(prev_tid) && prev_tid.size()>0) 
    {
      // Dispatches current tasks for execution
      //std::cerr << ts->getId() << "-------" << prev_tid<< std::endl;
      std::cerr << "  [" <<prev_tid << "]" ;
      if (gpuEngine)
        ts->run(ExecEngineConstants::GPU, 0);
      else 
        ts->run(ExecEngineConstants::CPU, 0);
      ts->~JoinTask();

      ts = new JoinTask(JCARDINALITY);
    }
    // actual geometry info: did,oid,num_ver,mbb, geom
    int i = input_line[pos+1] - '1'; // array position 
    pos2=input_line.find_first_of(COMMA,pos+3); //oid = input_line.substr(pos+3,pos2-pos-3) 
    pos=input_line.find_first_of(COMMA,pos2+1); //num_ver = input_line.substr(pos2+1,pos)
    ts->nr_vertices[i] += std::stoi(input_line.substr(pos2+1,pos-pos2-1));
    ts->geom_arr[i]->push_back(input_line.substr(pos2+1)); // mbb, geom
    prev_tid = tid; 
  }
  //cerr "running" << endl;
  if (gpuEngine)
    ts->run(ExecEngineConstants::GPU, 0);
  else 
    ts->run(ExecEngineConstants::CPU, 0);
  ts->~JoinTask();

  //std::cerr << "  ["<<tid << "]" << std::endl;
  //return 0;


  fini_device_streams(1);
  gettimeofday(&t2, NULL);
  std::cout<< "Total exec time:  " <<DIFF_TIME(t1, t2) <<" s." <<endl;

  return 0;
}


