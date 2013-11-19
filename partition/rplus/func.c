#include <stdio.h>
#include <sys/types.h>
#include <sys/file.h>
#include <ctype.h>
#include "macros.h"
#include "options.h"
#include "index.h"
#include "global.h"
#include "assert.h"

extern	pageNo;
extern	struct Node	*SearchBuf[MAXLEVELS];

/*
 ** Create an index file with the given index name in the first argument. If 
 ** there exits this file, it will return FALSE, o/w return the file discriptor.
 */
int CreateIndex (char *indexName, struct Node *root)
{
    int		i, idxp;

    if (( idxp = open( indexName, O_RDWR )) > 0 )
    {
	printf("The index file %s already exists\n", indexName );
	close( idxp );
	return FALSE;
    }
    else
    {
	printf("The new index \"%s\" is created.\n", indexName );

	idxp = creat( indexName, 0755 );
	close( idxp );
	idxp = open( indexName, O_RDWR );
	InitNode( root );
	root->level = 0;
	for (i=0; i<NUMDIMS; i++) {
	    root->rect.boundary[NUMDIMS+i] = MAXINT;
	    root->rect.boundary[i] = MININT;
	}
	PutOnePage( idxp, 0, root );
	pageNo = 2;
    }
    return idxp;
}

int DropIndex(char* indexName)
{
    char s[100];

    strcpy( s, "/bin/rm -f " );
    strcat( s, indexName );
    system( s );
}

    int 
OpenIndex( indexName, root )
    char		*indexName;
    struct Node	**root;
{
    struct Node	*GetOnePage();
    int		idxp;

    if (( idxp = open( indexName, O_RDWR )) < 0 )
    {
	printf("The index %s does not exist. Create it first.\n", indexName );
	close( idxp );
	return FALSE;
    }
    else
	(*root) = GetOnePage( idxp, 0 );
    return idxp;
}


CloseIndex( idxp, root )
    int		idxp;
    struct Node	*root;
{
    PutOnePage( idxp, 0, root );
    close( idxp );
}

int GetOneRect(FILE* fp, struct Rect* rect)
{
    int	in;
    float x1, y1, x2, y2;
    int j;

    j = fscanf( fp, "%d %e %e %e %e", &in, &x1, &y1, &x2, &y2 );
    if (j == EOF)
	return 0;
    rect->boundary[0] = (int) x1;
    rect->boundary[1] = (int) y1;
    rect->boundary[2] = (int) x2;
    rect->boundary[3] = (int) y2;
    return ( 0 - in );
}

int GetManyRect(FILE* fp, struct Rect** rect, int* id)
{
    int i=0;

    while ( (id[i] = GetOneRect(fp,rect[i])) != 0)
    {
	i++;
	if (i == NODECARD)
	    break;
    }
    return (i);
}

    int 
InsertOneRect( idxp, root, rect, id )
    int		idxp;
    struct Node	**root;
    register struct	Rect *rect;
    int		id;
{
    assert( id < 0 );
    /****** rect id is not saved temporarily *******
      return( InsertRect( idxp, rect, id, root, 0 ));
     **********************************************/
    return( InsertRect( idxp, rect, id, root, 0 ));
}

int InsertManyRect(
	int		idxp,
	struct Node	**root,
	struct Rect	**rect,
	int		*id, 
	int num)
{
    int	j;

    for (j=0;j<num;j++)
	assert( id[j] < 0 );
    /****** rect id is not saved temporarily *******
      return( Pa_InsertRect( idxp, rect, id, root, 0, num ));
     **********************************************/
    /*	return( Pa_InsertRect( idxp, rect, root, 0, num )); */
}

int PackInput(int idxp, struct Node** root,char* fileName)
{
    FILE		*fp;
    struct Rect	*rect[NODECARD];
    int		num, id[NODECARD];
    int i,j;

    if ((fp = fopen( fileName, "r" )) == NULL )
    {
	printf("There is no input file %s\n", fileName );
	return FALSE;
    }

    for (i=0; i<NODECARD; i++)
	rect[i] = (struct Rect *) myalloc( sizeof (struct Rect) );

    while (1)
    {
	num = GetManyRect( fp, rect, id) ;

	if (num) {

#ifdef PRINT
	    for (j=0; j<num; j++)
		printf ("Inserting ID = %d\n",id[j]);
#endif
	    InsertManyRect( idxp, root, rect, id,num );
	}
	else
	    break;
    }

    return TRUE;
}

int NoPackInput(int idxp, struct Node	**root, char* fileName)
{
    FILE		*fp;
    struct Rect	*rect;
    int		id;

    if ((fp = fopen( fileName, "r" )) == NULL )
    {
	printf("There is no input file %s\n", fileName );
	return FALSE;
    }

    rect = (struct Rect *) myalloc( sizeof (struct Rect) );

    while ( id = GetOneRect( fp, rect ) ) {
#ifdef PRINT
	printf ("Inserting ID = %d\n",id);
	PrintRect( rect );
#endif
	InsertOneRect( idxp, root, rect, id );
    }

    return TRUE;
}

    int
BatchDelete( idxp, root, fileName )
    int		idxp;
    struct Node	**root;
    char		*fileName;
{
    FILE *fp;
    struct Rect *rect;

    if ((fp = fopen( fileName, "r" )) == NULL ) {
	printf("There is no input file %s\n", fileName );
	return FALSE;
    }   

    rect = (struct Rect *) myalloc( sizeof (struct Rect) );
    while ( GetOneRect( fp, rect ) )
    {
	printf ("\n");
	printf ("DELETE Rectangle -----\n");
	PrintRectIdent( rect );
	DeleteOneRect( idxp, root, rect ) ;
    }
    return TRUE;
}

    int 
BatchSearch( idxp, root, fileName, rel, mode )
    int		idxp; 
    struct Node	*root; 
    char		*fileName; 
    char		*rel; 		/* topological or direction relation */
    char		*mode; 		/* mode of relation: OBJ / MBR */
{ 
    register int	i, j;
    FILE	*fp; 
    struct Rect *rect;

    if ((fp = fopen( fileName, "r" )) == NULL )
    {
	printf("There is no input file %s\n", fileName ); 
	return FALSE; 
    }    

    rect = (struct Rect *) myalloc( sizeof (struct Rect) );
    while ( GetOneRect( fp, rect ) )
    {
	SearchCount++;
	/* YANNIS
	   printf ("SEARCH %d\n",SearchCount);
	   printf ("SEARCH Rectangle -----\n");
	   PrintRectIdent(rect);
	   */

	for (i=0; i<MAXLEVELS; i++)
	    SearchBuf[i] = (struct Node *) myalloc( sizeof (struct Node) );

	j = SearchOneRect( idxp, root, rect, rel, mode );
	HitCount = HitCount + j;

	for (i=0;i<MAXLEVELS;i++)
	    myfree(SearchBuf[i]);
    }

    return TRUE;
}
