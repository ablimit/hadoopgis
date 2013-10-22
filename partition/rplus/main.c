/* main.c
||
|| Description:
||	A test main program to "batch load"  w/out packing an r-tree
||	and search into the r-tree using a window file
||
|| Parameters:
||	1 - raw data file to load from OR window file
||	2 - r-tree "base name" (.idx and .dat suffixes will be added)
||	3 - version (1:insert, 2:search)
||	4 - topological/direction relation
||	5 - mode of search (OBJ or MBR)
*/

#include	<stdio.h>
#include	<ctype.h>
#include	<sys/file.h>

#include	"macros.h"
#include	"options.h"
#include	"assert.h"
#include	"index.h"
#include	"global.h"

int     NO_REFIN;

/* Global Crap */
FILE	*fp;
int	pageNo, MinFill, PackFlag;
struct	Node	*SearchBuf[MAXLEVELS];
	int magicNum;


main(argc, argv)
int argc;
char *argv[];
{
	int idxp;
	char	ixfnam[MAXNAMLEN];

	if (argc != 6)
	{
		fprintf(stderr, "Usage: rpt <raw data file> <r-tree file> <version [1,2] > <top/dir rel.> <OBJ/MBR>\n");
		exit(1);
	}

	/* Form the index file name */
	strcpy(ixfnam, argv[2]);
	strcat(ixfnam, IXSUFFIX);

	if (strcmp(argv[3],"1") == 0)	/* INSERT */
	{
	Initialize();

	if ( (idxp = CreateIndex(ixfnam, Root)) == 0 )
		idxp = OpenIndex(ixfnam, &Root);

	MinFill = (int) NODECARD/2;
	PackFlag = 0;

	ResetClock();
	StartClock();
    	NoPackInput( idxp, &Root, argv[1] );
	StopClock();
	InsertStats();
fflush(stdout);
	CloseIndex(idxp, Root );
	}

	else if (strcmp(argv[3],"2") == 0) /* SEARCH */
	{
	idxp = OpenIndex(ixfnam, &Root);

	NO_REFIN = 0;

	ResetClock();
	StartClock();
    	BatchSearch( idxp, Root, argv[1], argv[4], argv[5] );
	StopClock();
	SearchStats();

	CloseIndex(idxp, Root );
	}

	else
	{
		fprintf(stderr, "ERROR: version should be 1 or 2\n");
		exit(1);
	}

	exit(0);
}
    
    
Initialize()
{
	register int i;

	StatFlag = 1;

	for (i=0; i< NUMDIMS; i++)
	{
		CoverAll.boundary[i] = MININT;
		CoverAll.boundary[NUMDIMS+i] = MAXINT;
	}

	Root = (struct Node *) myalloc(sizeof(struct Node));

	LeafCount = 1;
	NodeCount = 1;

	return;
}
