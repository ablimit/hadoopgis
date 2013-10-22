/* macros.h
||
||	This header file contains generic constant and macro definitions for
|| the r-tree library.
*/

//#ifndef	NULL
//#define NULL 0
//#endif

#define EOS '\0'

#define True 1
#define TRUE 1
#define False 0
#define FALSE 0
#define UNKNOWN 0

#define And &&
#define Or ||
#define Not !

#define elif else if
#define loop for(;;)

#define PI 3.141592654
#define DEGTORAD ((2*PI)/360)
#define RADTODEG (360/(2*PI))
#define INTOCM	2.54
#define CMTOIN	0.39370079

#define MAXINT 1000000
#define MININT 0

/* #define MAXINT 0x7fffffff */
/*  2147483647 */
/* #define MININT 0x80000000 */
/* -2147483648 */

/* Comparison Macros */
#define max(a,b) ((a) > (b) ? (a) : (b))
#define min(a,b) ((a) < (b) ? (a) : (b))
#define abs(a) ((a) >= 0 ? (a) : (-(a)))
#define SwapInts(a,b) {int rt_c; rt_c = a; a = b; b = rt_c;}
