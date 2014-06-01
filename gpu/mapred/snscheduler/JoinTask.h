

#ifndef TASKID_H_
#define TASKID_H_

#include "Task.h"

class JoinTask: public Task {
private:
 int N ;

public:
 vector<string>** geom_arr = NULL;
 vector<int> nr_vertices;
	
 JoinTask(int n);
	
	virtual ~JoinTask();

	bool run(int procType=ExecEngineConstants::CPU, int tid=0);
};

#endif /* TASKID_H_ */
