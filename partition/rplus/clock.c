/*
|| clock.c
||
|| Functions:
||	ResetClock()
||	StartClock()
||	StopClock()
*/

#include	<sys/types.h>
#include	<sys/times.h>
#include 	"options.h"
#include 	"index.h"


struct	tms start, stop, net;

extern	float	UserTime, SystemTime;
extern	long	ElapsedTime;
long	startElapsed, stopElapsed;

ResetClock()
{
	UserTime = SystemTime = 0;
	ElapsedTime = 0;

	net.tms_utime = net.tms_stime = net.tms_cutime = net.tms_cstime = 0;
}

StartClock()
{
	long time();

	times(&start);
	startElapsed = time(0);
}

StopClock()
{
	long time();

	times(&stop);
	stopElapsed = time(0);

	net.tms_utime +=  stop.tms_utime -  start.tms_utime;
	net.tms_stime +=  stop.tms_stime -  start.tms_stime;
	net.tms_cutime += stop.tms_cutime - start.tms_cutime;
	net.tms_cstime += stop.tms_cstime - start.tms_cstime;

	UserTime = (float)net.tms_utime / 60;
	SystemTime = (float)net.tms_stime / 60;

	ElapsedTime += stopElapsed - startElapsed;
}
