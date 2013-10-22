#include <stdio.h>
#include "macros.h"
#include "options.h"
#include "index.h"
#include "global.h"
#include "assert.h"

/*-----------------------------------------------------------------------------
| Space allocation routines. - just keep track of No. of allocations
|
| Call myalloc() like malloc().
| Call myfree() like free().
-----------------------------------------------------------------------------*/

static	int	puse=0; vuse=0;
static	int	all = 0;
static	int	maxuse=0;

char *
myalloc(n)
register int	 n;
{
	char	*malloc();

	all++;
	puse++;

	if ((puse-vuse) > maxuse)
		maxuse++;

	return malloc(n);
}

myfree(p)
char	*p;
{
	all--;
	vuse++;

	free(p);
}

myfreeOFN( p )
register struct OverFlowNode 	*p;
{
    	register struct OverFlowNode 	*q;
    	while ( p != NULL ) {
		all --;
		vuse++;
		q = p->next2;
		free(p);
		p = q;
    	}
}
