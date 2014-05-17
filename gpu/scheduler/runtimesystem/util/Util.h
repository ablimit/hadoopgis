/*
 * Util.h
 *
 *  Created on: Feb 15, 2012
 *      Author: george
 */

#ifndef UTIL_H_
#define UTIL_H_

#include <sys/time.h>

class Util {
private:
	Util();
	virtual ~Util();
public:
	static long long ClockGetTime();

};

#endif /* UTIL_H_ */
