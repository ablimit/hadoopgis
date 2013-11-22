/* #define DEBUG 1 */
# include <stdio.h>
#include <sys/types.h>
#include <sys/file.h>
#include "macros.h"
#include "options.h"
#include "index.h"
#include "global.h"
#include "assert.h"

extern int	pageNo;

#ifdef DEBUG
	extern int magicNum;
#endif

struct Node *
GetOnePage( idxp, offset )
int idxp;
off_t offset;
{
    struct Node *node;
    char ws[PAGESIZE];
    short int level;
    int n;

#ifdef DEBUG
	printf("I TRY TO GET A LEAF/NONLEAF PAGE AT OFFSET %d\n",offset);
	printf("GLOBAL pageNo = %d\n",pageNo);
#endif
    assert( offset >= 0 );
    node = (struct Node *) myalloc (sizeof(struct Node));
    lseek( idxp, offset*LPAGESIZE, L_SET );
    n = read( idxp, ws, PAGESIZE );
    bcopy(ws+COUNT_SIZE,(char *) &level, LEVEL_SIZE);

    if (level == 0) RReadOneLPage( idxp, ws, node, offset);
    else RReadOnePage( idxp, ws, node, offset);
    return node;
}

int
ReadOnePage( idxp,node,offset,level )
int		idxp;
struct Node	*node;
off_t		offset;
int	level;
{
    char ws[PAGESIZE];
    int n;

    assert( offset >= 0 );
    assert(node);
    lseek( idxp, offset*LPAGESIZE, L_SET );
    n = read( idxp, ws, PAGESIZE );

    if (level == 0) RReadOneLPage( idxp, ws, node, offset);
    else RReadOnePage( idxp, ws, node, offset);

    return ;
}

int
RReadOnePage( idxp,ws,node,offset )
int		idxp;
char		*ws;
struct Node	*node;
off_t		offset;
{
    int n;
    register int i, j;
    int d1, d2;

#ifdef DEBUG
    int CheckFullCover();
    int mmm=1;
    struct Branch ba[NODECARD+1];

    if (offset == magicNum)
        mmm = 1;
#endif
#ifdef DEBUG
	printf("\tI TRY TO READ A PAGE AT OFFSET %d\n",offset);
#endif

    assert( offset >= 0 );
    node->pageNo = offset;
    bcopy(ws,(char *) &(node->count), COUNT_SIZE);
    bcopy(ws+COUNT_SIZE,(char *) &(node->level), LEVEL_SIZE);

    for( i=0; i<NUMDIMS; i++ ) {
        d1 = HEAD_SIZE + i * COORD_SIZE;
        d2 = d1 + COORD_SIZE * NUMDIMS;
	bcopy(ws+d1,(char *) &(node->rect.boundary[i]), COORD_SIZE);
	bcopy(ws+d2,(char *) &(node->rect.boundary[i+NUMDIMS]), COORD_SIZE);
    }
    for ( i=0; i<NODECARD; i++ ) {
        for( j=0; j<NUMDIMS; j++ ) {
            d1 = HEAD_SIZE + COORD_SIZE * NUMSIDES + i * BRANCHSIZE + j * POINTER_SIZE;
            d2 = d1 + COORD_SIZE * NUMDIMS;
	    bcopy(ws+d1,(char *) &(node->branch[i].minrect.boundary[j]), COORD_SIZE);
	    bcopy(ws+d2,(char *) &(node->branch[i].minrect.boundary[j+NUMDIMS]), COORD_SIZE);
        }
        for( j=0; j<NUMDIMS; j++ ) { 
            d1 = HEAD_SIZE + 2 * COORD_SIZE * NUMSIDES + i * BRANCHSIZE + j * POINTER_SIZE;
            d2 = d1 + COORD_SIZE * NUMDIMS;
	    bcopy(ws+d1,(char *) &(node->branch[i].rect.boundary[j]), COORD_SIZE);
	    bcopy(ws+d2,(char *) &(node->branch[i].rect.boundary[j+NUMDIMS]), COORD_SIZE);
        }
        d1 = HEAD_SIZE + 3 * COORD_SIZE * NUMSIDES + i * BRANCHSIZE; 
	bcopy(ws+d1,(char *) &(node->branch[i].son), POINTER_SIZE);
    }
    if ( pageNo == 0 )  /* the reading page must be the root node */
	bcopy(ws+PAGESIZE-sizeof(int) ,(char *) &pageNo, sizeof(int));

#ifdef DEBUG
    if ( node->level > 0 ) {
        if (CheckFullCover( node ) == FALSE)  {
	    ba[0].rect = node->rect;
            for (i=0; i<NODECARD; i++)
                if (node->branch[i].son >= 0)
                    ba[mmm++].rect = node->branch[i].rect;
/*            draw( ba, mmm ); */
        }
    }
#endif
#ifdef DEBUG
	printf("\tI READ PAGE NUMBER %d LEVEL=%d AT OFFSET %d\n",node->pageNo,node->level,offset);
#endif

    return ;
}

int
RReadOneLPage( idxp,ws,node,offset )
int		idxp;
char		*ws;
struct Node	*node;
off_t		offset;
{
    int n;
    register int i, j;
    int d1, d2;
    struct Rect IntersectRect();

#ifdef DEBUG
    int CheckFullCover();
    int mmm=1;
    struct Branch ba[NODECARD+1];

    if (offset == magicNum)
        mmm = 1;
#endif

#ifdef DEBUG
	printf("\tI TRY TO READ A LEAF PAGE AT OFFSET %d\n",offset);
#endif
    assert( offset >= 0 );
    assert(node);
    node->pageNo = offset;
    bcopy(ws,(char *) &(node->count), COUNT_SIZE);
    bcopy(ws+COUNT_SIZE,(char *) &(node->level), LEVEL_SIZE);

    for( i=0; i<NUMDIMS; i++ ) {
        d1 = HEAD_SIZE + i * COORD_SIZE;
        d2 = d1 + COORD_SIZE * NUMDIMS;
	bcopy(ws+d1,(char *) &(node->rect.boundary[i]), COORD_SIZE);
	bcopy(ws+d2,(char *) &(node->rect.boundary[i+NUMDIMS]), COORD_SIZE);
    }
    for ( i=0; i<NODECARD; i++ ) {
        for( j=0; j<NUMDIMS; j++ ) {
            d1 = HEAD_SIZE + COORD_SIZE * NUMSIDES + i * LBRANCHSIZE + j * POINTER_SIZE;
            d2 = d1 + COORD_SIZE * NUMDIMS;
	    bcopy(ws+d1,(char *) &(node->branch[i].rect.boundary[j]), COORD_SIZE);
	    bcopy(ws+d2,(char *) &(node->branch[i].rect.boundary[j+NUMDIMS]), COORD_SIZE);
        }
	node->branch[i].minrect = IntersectRect(&node->branch[i].rect,&node->rect);

        d1 = HEAD_SIZE + 2 * COORD_SIZE * NUMSIDES + i * LBRANCHSIZE; 
	bcopy(ws+d1,(char *) &(node->branch[i].son), POINTER_SIZE);
    }
    if ( pageNo == 0 )  /* the reading page must be the root node */
	bcopy(ws+LPAGESIZE-sizeof(int) ,(char *) &pageNo, sizeof(int));

#ifdef DEBUG
    if ( node->level > 0 ) {
        if (CheckFullCover( node ) == FALSE)  {
	    ba[0].rect = node->rect;
            for (i=0; i<NODECARD; i++)
                if (node->branch[i].son >= 0)
                    ba[mmm++].rect = node->branch[i].rect;
/*            draw( ba, mmm ); */
        }
    }
#endif

#ifdef DEBUG
	printf("\tI READ LEAF PAGE NUMBER %d LEVEL=%d AT OFFSET %d\n",node->pageNo,node->level,offset);
#endif

    return ;
}

void PutOnePage ( int idxp, off_t offset,struct Node *node)
{
    if (node->level == 0) RPutOneLPage(idxp,offset,node);
    else RPutOnePage(idxp,offset,node);
}

void RPutOnePage (int idxp, off_t offset, struct Node *node)
{
    char ws[PAGESIZE];
    register int i, j;
    int d1, d2;

#ifdef DEBUG
    int CheckFullCover();
    int mmm=1;
    struct Branch ba[NODECARD+1];

    if (offset == magicNum)
        mmm = 1;
    if (node->pageNo != offset)
        mmm = 1;
    
    if (node->level > 0) {
       if (CheckFullCover( node ) == FALSE)  {
/*	    PrintNode(node); */
	    printf("not full   ERROR  ERROR    ERROR    ERROR.\n");
            ba[0].rect = node->rect;
            for (i=0; i<NODECARD; i++) 
	        if (node->branch[i].son >= 0) 
	            ba[mmm++].rect = node->branch[i].rect;
/*            draw( ba, mmm ); */
        }
    }
#endif

    bcopy((char *) &(node->count),ws, COUNT_SIZE);
    bcopy((char *) &(node->level),ws+COUNT_SIZE, LEVEL_SIZE);

    for( i=0; i<NUMDIMS; i++ ) {
        d1 = HEAD_SIZE + i * COORD_SIZE;
        d2 = d1 + COORD_SIZE * NUMDIMS;
	bcopy((char *) &(node->rect.boundary[i]),ws+d1, COORD_SIZE);
	bcopy((char *) &(node->rect.boundary[i+NUMDIMS]),ws+d2, COORD_SIZE);
    }
    for ( i=0; i<NODECARD; i++ ) {
        for( j=0; j<NUMDIMS; j++ ) {
            d1 = HEAD_SIZE + COORD_SIZE * NUMSIDES + i * BRANCHSIZE + j * POINTER_SIZE;
            d2 = d1 + COORD_SIZE * NUMDIMS;
	    bcopy((char *) &(node->branch[i].minrect.boundary[j]),ws+d1, COORD_SIZE);
	    bcopy((char *) &(node->branch[i].minrect.boundary[j+NUMDIMS]),ws+d2, COORD_SIZE);
        }
        for( j=0; j<NUMDIMS; j++ ) { 
            d1 = HEAD_SIZE + 2 * COORD_SIZE * NUMSIDES + i * BRANCHSIZE + j * POINTER_SIZE;
            d2 = d1 + COORD_SIZE * NUMDIMS;
	    bcopy((char *) &(node->branch[i].rect.boundary[j]),ws+d1, COORD_SIZE);
	    bcopy((char *) &(node->branch[i].rect.boundary[j+NUMDIMS]),ws+d2, COORD_SIZE);
        }
        d1 = HEAD_SIZE + 3 * COORD_SIZE * NUMSIDES + i * BRANCHSIZE; 
	bcopy((char *) &(node->branch[i].son),ws+d1, POINTER_SIZE);
    }
    if ( offset == 0 )  /* the reading page must be the root node */
	bcopy((char *) &pageNo, ws+PAGESIZE-sizeof(int) , sizeof(int));

    lseek( idxp, offset*LPAGESIZE, L_SET ); 
    write( idxp, ws, PAGESIZE );
#ifdef DEBUG
	printf("\tI WROTE PAGE NUMBER %d LEVEL=%d AT OFFSET %d\n",node->pageNo,node->level,offset);
#endif
}
       
void RPutOneLPage (int idxp, off_t offset, struct Node *node)
{
    char ws[LPAGESIZE];
    int i, j;
    int d1, d2;

#ifdef DEBUG
    int CheckFullCover();
    int mmm=1;
    struct Branch ba[NODECARD+1];

    if (offset == magicNum)
        mmm = 1;
    if (node->pageNo != offset)
        mmm = 1;
    
    if (node->level > 0) {
       if (CheckFullCover( node ) == FALSE)  {
/*	    PrintNode(node); */
	    printf("not full   ERROR  ERROR    ERROR    ERROR.\n");
            ba[0].rect = node->rect;
            for (i=0; i<NODECARD; i++) 
	        if (node->branch[i].son >= 0) 
	            ba[mmm++].rect = node->branch[i].rect;
/*            draw( ba, mmm ); */
        }
    }
#endif

    bcopy((char *) &(node->count),ws, COUNT_SIZE);
    bcopy((char *) &(node->level),ws+COUNT_SIZE, LEVEL_SIZE);

    for( i=0; i<NUMDIMS; i++ ) {
        d1 = HEAD_SIZE + i * COORD_SIZE;
        d2 = d1 + COORD_SIZE * NUMDIMS;
	bcopy((char *) &(node->rect.boundary[i]),ws+d1, COORD_SIZE);
	bcopy((char *) &(node->rect.boundary[i+NUMDIMS]),ws+d2, COORD_SIZE);
    }
    for ( i=0; i<NODECARD; i++ ) {
        for( j=0; j<NUMDIMS; j++ ) {
            d1 = HEAD_SIZE + COORD_SIZE * NUMSIDES + i * LBRANCHSIZE + j * POINTER_SIZE;
            d2 = d1 + COORD_SIZE * NUMDIMS;
	    bcopy((char *) &(node->branch[i].rect.boundary[j]),ws+d1, COORD_SIZE);
	    bcopy((char *) &(node->branch[i].rect.boundary[j+NUMDIMS]),ws+d2, COORD_SIZE);
        }
        d1 = HEAD_SIZE + 2 * COORD_SIZE * NUMSIDES + i * LBRANCHSIZE; 
	bcopy((char *) &(node->branch[i].son),ws+d1, POINTER_SIZE);
    }
    if ( offset == 0 )  /* the reading page must be the root node */
	bcopy((char *) &pageNo, ws+LPAGESIZE-sizeof(int) , sizeof(int));

    lseek( idxp, offset*LPAGESIZE, L_SET ); 
    write( idxp, ws, LPAGESIZE );
#ifdef DEBUG
	printf("\tI WROTE LEAF PAGE NUMBER %d LEVEL=%d AT OFFSET %d\n",node->pageNo,node->level,offset);
#endif
}
       
struct OverFlowNode * GetOverFlowPage (int idxp, off_t offset)
{
    int n;
    char ws[LPAGESIZE];
    register int i, j, myPageNo;
    struct OverFlowNode *overFlowNode, *rp;
    int OFFsize;
    int d1, d2;

#ifdef DEBUG
	printf("I TRY TO READ A OVERF PAGE AT OFFSET %d\n",offset);
    if (offset == magicNum)
        n=0;
#endif

    OFFsize = sizeof(int);
    overFlowNode = (struct OverFlowNode *) myalloc(sizeof(struct OverFlowNode));
    InitOFNode( overFlowNode );
    overFlowNode-> pageNo = offset;
    myPageNo = offset;
    rp = overFlowNode;
    do {
	rp->pageNo = offset;
        lseek( idxp, offset*LPAGESIZE, L_SET );
        n = read( idxp, ws, LPAGESIZE );
        bcopy(ws,(char *) &(rp->count), COUNT_SIZE);
        bcopy(ws+COUNT_SIZE,(char *) &offset, OFFsize);
        for (i=0; i<rp->count; i++)  {
	    for (j=0; j<NUMDIMS; j++) {
	    	d1 = COUNT_SIZE + OFFsize + i * NUMSIDES * COORD_SIZE + j * COORD_SIZE;
	    	d2 = d1 + COORD_SIZE * NUMDIMS;
		bcopy(ws+d1,(char *) &(rp->rect[i].boundary[j]), COORD_SIZE);
		bcopy(ws+d2,(char *) &(rp->rect[i].boundary[j+NUMDIMS]), COORD_SIZE);
	    }
        }
	if (offset == myPageNo) 
	    return overFlowNode;
	else {
	    myPageNo = offset;
	    rp->next2=(struct OverFlowNode*)myalloc(sizeof(struct OverFlowNode));
	    rp = rp->next2;
	    InitOFNode( rp );
  	}
    }
    while (offset != rp->pageNo);
    return overFlowNode;
}


void PutOverFlowPage (int idxp, struct OverFlowNode *overFlowNode)
{
    off_t offset;
    char ws[LPAGESIZE];
    register int i, j;
    int tempNo, d1, d2;
    int OFFsize;
    struct OverFlowNode * rp;

#ifdef DEBUG
	printf("I TRY TO WRITE A OVERF PAGE AT OFFSET %d\n",offset);
    if (overFlowNode->pageNo == magicNum )
        i=0;
#endif

    OFFsize = sizeof(int);
    rp = overFlowNode;

    while (rp != NULL) {
    	bcopy((char *) &(rp->count),ws, COUNT_SIZE);
        if (rp->next2 == NULL)
	    if (rp->pageNo == UNKNOWN) {
		offset = pageNo;
		rp->pageNo = pageNo;
		pageNo++ ;
    		bcopy((char *) &pageNo,ws+COUNT_SIZE, sizeof(int));
	    }
	    else {
		offset = rp->pageNo;
    		bcopy((char *) &(rp->pageNo),ws+COUNT_SIZE, sizeof(int));
	    }
	else if (rp->pageNo == UNKNOWN) {
	    	rp->pageNo = pageNo;
	    	offset = pageNo;
		tempNo = pageNo + 1;
    		bcopy((char *) &tempNo,ws+COUNT_SIZE, sizeof(int));
   	}
	else if (rp->next2->pageNo == UNKNOWN) {
	    	offset = rp->pageNo;
    		bcopy((char *) &pageNo,ws+COUNT_SIZE, sizeof(int));
	}
  	else {
	    	offset = rp->pageNo;
    		bcopy((char *) &(rp->next2->pageNo),ws+COUNT_SIZE, sizeof(int));
	}
        for (i=0; i<rp->count; i++) {
	    for (j=0; j<NUMDIMS; j++) {
	    	d1 = COUNT_SIZE + OFFsize + i * NUMSIDES * COORD_SIZE + j * COORD_SIZE;
	    	d2 = d1 + NUMDIMS * COORD_SIZE;
		bcopy((char *) &(rp->rect[i].boundary[j]),ws+d1, COORD_SIZE);
		bcopy((char *) &(rp->rect[i].boundary[j+NUMDIMS]),ws+d2, COORD_SIZE);
	    }
	}
        lseek( idxp, offset*LPAGESIZE, L_SET );
        write( idxp, ws, LPAGESIZE );
	rp = rp->next2;
    }
}

double AArea(struct Rect *rect)
{

        register int i;
	register double area;

        area = 1.0;
        for (i=0; i<NUMDIMS; i++)
                area *= ((double) rect->boundary[i+NUMDIMS] - (double) rect->boundary[i]);
        return area;
}

 /* Rectangle r1 is inside rectangle r2 */
int Within(struct Rect r1, struct Rect r2)
{
    int tmp;

    if (r1.boundary[0] > r1.boundary[2]) {
            tmp = r1.boundary[0];
            r1.boundary[0] = r1.boundary[2];
            r1.boundary[2] = tmp;
    }
    if ((r1.boundary[0]>=r2.boundary[0]) && (r1.boundary[1]>=r2.boundary[1]) &&
        (r1.boundary[2]<=r2.boundary[2]) && (r1.boundary[3]<=r2.boundary[3]) )
        return TRUE;
    else return FALSE;
}


int OverLaps(struct Rect r1, struct Rect r2)
{
    int tmp;

    if (r1.boundary[0] > r1.boundary[2]) {
            tmp = r1.boundary[0];
            r1.boundary[0] = r1.boundary[2];
            r1.boundary[2] = tmp;
    }
    if (r2.boundary[0] > r2.boundary[2]) {
            tmp = r2.boundary[0];
            r2.boundary[0] = r2.boundary[2];
            r1.boundary[2] = tmp;
    }
    return( Overlap( &r1, &r2) );
}


int CheckFullCover(struct Node *node)
{
    int i, j, k;
    double area, AArea();
    int X1[NODECARD];
    int X2[NODECARD];
    int Y1[NODECARD];
    int Y2[NODECARD];
    int count;
    
    area = 0;
    count = 0;

    for (i=0; i<NODECARD; i++) {
        if (node->branch[i].son >= 0) {
	    X1[count] = node->branch[i].rect.boundary[0];
	    X2[count] = node->branch[i].rect.boundary[2];
	    Y1[count] = node->branch[i].rect.boundary[1];
	    Y2[count++] = node->branch[i].rect.boundary[3];
            if (Within(node->branch[i].rect, node->rect) == FALSE)
	       {
		printf("NOT WITHIN\n");
                return FALSE;
	       }

            area += AArea( &node->branch[i].rect );
            for (j=i+1; j<NODECARD; j++)
                if ((node->branch[j].son >0) &&
                   (OverLaps( node->branch[i].rect, node->branch[j].rect))) {
                    printf("over lap VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV\n");
                    PrintRect( &node->branch[i].rect );
                    PrintRect( &node->branch[j].rect );
                    printf("over lap ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n");
                    return FALSE;
                }
        }
    }
    if (AArea(&node->rect) == area) return TRUE;
    else
    {
    printf ("NOT EQUAL AREAS %f\t%f\n",AArea(&node->rect),area);
    for (i=0; i<count; i++) 
	printf ("%d %d\t\t%d %d\n",X1[i],X2[i],Y1[i],Y2[i]);
    printf ("\n");
    return FALSE;
    }
}
