/* options.h
||
|| This header file contains definitions of constants which may be altered
|| to change the charastics of the r-rtrees.
*/

/* Un-coment this constant to turn on debug printing */
/* #define PRINT 1 */
/* #define DEBUG 1 */

#define LPAGESIZE	1024
#define PAGESIZE	2*LPAGESIZE
#define NUMDIMS		2	/* number of dimensions */
#define NUMSIDES	2*NUMDIMS
#define BRANCHSIZE	(int)sizeof(struct Branch)
#define LBRANCHSIZE	(int)sizeof(struct LBranch)

/* branching factor of a node */
/* this is wrong #define NODECARD	(int)((PAGESIZE-LEVEL_SIZE-COUNT_SIZE-HEAD_SIZE-sizeof(struct Rect))/BRANCHSIZE) */
#define LNODECARD	(int)((LPAGESIZE-LEVEL_SIZE-COUNT_SIZE-HEAD_SIZE)/LBRANCHSIZE)
#define NODECARD LNODECARD

#define Thresh (int)(NODECARD/2)
#define OVERFLOWNODECARD (int)((LPAGESIZE-COUNT_SIZE-sizeof(int))/sizeof(struct Rect))

#define FILE_PATH	""
#define HEAD_SIZE	4
#define COUNT_SIZE	2
#define LEVEL_SIZE	2
#define MAXLEVELS	5
#define POINTER_SIZE	sizeof(int)
#define COORD_SIZE	sizeof(int)
#define MAXAREA 4.61168601413242061e+18


/* Define constants for index file names */
#define	MAXNAMLEN	40
#define	IXSUFFIX	".idx"
#define	DATASUFFIX	".dat"
