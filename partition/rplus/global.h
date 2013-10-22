/* global.h
||
|| 	This header file contains r-tree global variable definitions.
*/

/* index root node pointer */
struct Node *Root;

/* rectangle covering all data read */
struct Rect	CoverAll;

/* balance criterion for node splitting */
/* int MinFill; */

/* the whole permited area */
struct Rect MaxArea;

/* rp: Head of list of rectangles when packing the tree */
struct ListRect *ListRectHead;



/* times */
long	ElapsedTime;
float	UserTime, SystemTime;

/* variables for statistics */
int	StatFlag; /* tells if we are counting or not */
int	Deleting;

/* counters affected only when StatFlag set */
int	InsertCount;
int	DeleteCount;
int	ReInsertCount;
int	SplitCount;
int 	InSplitCount;
int 	DeSplitCount;
int 	DownwardSplitCount;
int 	CallCount;
float 	SplitMeritSum;
int	ElimCount;
int	EvalCount;
int	InTouchCount;
int	DeTouchCount;
int	SeTouchCount;

/* counters used even when StatFlag not set */
int	RectCount;
int	NodeCount;
int	LeafCount, NonLeafCount;
int	EntryCount;
int	SearchCount;
int	HitCount;

