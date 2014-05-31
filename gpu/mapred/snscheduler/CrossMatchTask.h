

#ifndef TASKID_H_
#define TASKID_H_

#include "Task.h"

class CrossMatch: public Task {
private:

public:
	CrossMatch();

	virtual ~CrossMatch();

	bool run(int procType=ExecEngineConstants::CPU, int tid=0);
};

#endif /* TASKID_H_ */
