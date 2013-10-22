/*
|| error.c
*/

#include	<stdio.h>
#include	<stdlib.h>

/* General purpose error routine.
** Prints a string and exits.
*/
void error(char *s)
{
	printf("*** %s\n", s);
	exit(1);
}
