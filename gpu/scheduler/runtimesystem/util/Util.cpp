/*
 * util.cpp
 *
 *  Created on: Feb 15, 2012
 *      Author: george
 */

#include "Util.h"

long long Util::ClockGetTime()
{
        struct timeval ts;
        gettimeofday(&ts, NULL);
        return (ts.tv_sec*1000000 + (ts.tv_usec))/1000LL;
}
