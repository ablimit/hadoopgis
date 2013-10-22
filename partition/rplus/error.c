/*
|| error.c
*/

#include	<stdio.h>

/* General purpose error routine.
** Prints a string and exits.
*/
error(s)
char *s;
{
	printf("*** %s\n", s);
	exit(1);
}
