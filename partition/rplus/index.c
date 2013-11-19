#include <stdio.h>
#include <stdlib.h>
#include "macros.h"
#include "options.h"
#include "index.h"
#include "global.h"
#include "assert.h"

extern	struct	Node	*SearchBuf[MAXLEVELS];

/* Make a new index, empty.  Consists of a single node.
*/
struct Node* NewIndex()
{
        struct Node *x, *NewNode();
        x = NewNode();
        x->level = 0; /* leaf */
        LeafCount++;
        return x;
}

/* Print out all the nodes in an index.
** For graphics output, displays lower boxes first so they don't obscure
** enclosing boxes.  For printed output, prints from root downward.
*/
void PrintIndex(int idxp, struct Node *n)
{
    int i, next;
    struct Node *nnn;
    struct Node *GetOnePage();
    assert(n);
    assert(n->level >= 0);

    /*********************ho***************
      PrintNode( n );
     *********************ho***************/
    if (n->level > 0)
    {
	for (i = 0; i < NODECARD; i++)
	{
	    if ((next = n->branch[i].son) > 0) {
		nnn = GetOnePage( idxp, next );
		PrintIndex(idxp, nnn);
		myfree( nnn );
	    }
	}
    }

}

/* Print out all the data rectangles in an index.
*/
PrintData(idxp, n)
    int idxp;
    struct Node *n;
{
    int i,next;
    struct Node *nnn;
    struct Node *GetOnePage();
    assert(n);
    assert(n->level >= 0);

    if (n->level == 0)
	TPrintNode(n);
    else
    {
	for (i = 0; i < NODECARD; i++)
	{
	    if ((next = n->branch[i].son) > 0) {
		nnn = GetOnePage( idxp, next );
		PrintData(idxp, nnn);
		myfree( nnn ); 
	    }
	}
    }
}

/* SearchOneRect in an index tree or subtree for all data retangles that
 ** overlap the argument rectangle.
 ** Returns the number of qualifying data rects.
 */
    int
SearchOneRect(idxp, n, r, rel, mode)
    int idxp;
    register struct Node *n;
    register struct Rect *r;
    char 	*rel;           /* topological or direction relation */
    char 	*mode;          /* mode of relation: OBJ / MBR */
{
    register int hitCount = 0;
    register int i, j;
    register struct OverFlowNode *over, *rp;
    struct OverFlowNode *GetOverFlowPage();

    assert(n);
    assert(n->level >= 0);
    assert(r);

    SeTouchCount++;



    /* YANNIS...
       Irel(idxp, r1, r2, rel, mode) : relation for intermediate nodes
       Trel(idxp, r1, r2, rel, mode) : relation for terminal nodes
       */

    if (n->level > 0) /* this is an internal node in the tree */
    {
	for (i=0; i<NODECARD; i++)
	    if (n->branch[i].son >= 0) 
		if ((n->level==1) && IsOverFlow(&n->branch[i].minrect)){
		    if (Irel(idxp, r, &n->branch[i].minrect, rel, mode)) {
			over = GetOverFlowPage( idxp, n->branch[i].son );
			rp = over;
			while ( rp != NULL ) {
			    for (j=0; j<rp->count; j++)
				if (Irel(idxp, r, &rp->rect[j], rel, mode)) {
				    /* printf("---->OVRFL Overlap with \n");
				       PrintRectIdent( &rp->rect[j] ); */
				}
			    rp = rp->next2;
			}
		    }
		    myfreeOFN( over );
		}
		else if (Irel(idxp, r, &n->branch[i].minrect, rel, mode)) {
		    ReadOnePage( idxp, SearchBuf[n->level], n->branch[i].son , (n->level)-1);
		    hitCount += SearchOneRect(idxp, SearchBuf[n->level], r, rel, mode);
		}
    }
    else /* this is a leaf node */
    {
	for (i=0; i<NODECARD; i++)
	{
	    if ((n->branch[i].son >= 0) 
		    && Trel(idxp, r, &n->branch[i].rect, rel, mode))
	    {
		hitCount++;
		/* printf("---->  Overlap with rect id %d\n", n->branch[i].son - 5000); */
		/* 				printf("---->  Overlap with \n");
						PrintRectIdent( &n->branch[i].rect ); */
	    }
	}
    }
    return hitCount;
}

/* Delete a data rectangle from an index structure.
 ** Pass in a pointer to a Rect, the r of the record, ptr to ptr to root node.
 ** Returns 1 if record not found, 0 if success.
 ** DeleteRect provides for eliminating the root.
 */
    int
DeleteOneRect(idxp, nn, r)
    int idxp;
    register struct Node **nn;
    register struct Rect *r;
{
    register int i;
    register struct Node *t;

    assert(r && nn );

    Deleting = TRUE;

#       ifdef PRINT
    printf("DeleteRect\n");
    PrintRect(r);
#       endif

    if (!DeleteRect2(idxp, r, *nn))
    {
	/* found and deleted a data item */
	if (StatFlag)
	    DeleteCount++;
	RectCount--;

	/* reinsert any branches from eliminated nodes */
	/* Non-leaf nodes not deleted in this version of R+ -trees */

	Deleting = FALSE;
	printf("---> The rectangle is deleted.\n" );
	return 1;
    }
    else
    {
	Deleting = FALSE;
	printf("---> The rectangle is not in R* tree.\n" );
	return 0;
    }
}

/* Delete a rectangle from non-root part of an index structure.
 ** Called by DeleteOneRect.  Descends tree recursively,
 ** merges branches on the way back up.
 */
    int
DeleteRect2(idxp, r, n)
    int idxp;
    register struct Rect *r;
    register struct Node *n;
{
    register int i, de;
    register int deleted = FALSE;
    register struct OverFlowNode *over;
    struct OverFlowNode *GetOverFlowPage();
    struct Node *newnode;
    int SOverlap();
    struct Rect MinNodeCover(), MinOFNodeCover();
    struct Node *GetOnePage(), *nnn;

    assert(r && n );
    assert(n->level >= 0);

    if (StatFlag)
	DeTouchCount++;

    if (n->level > 0) /* not a leaf node */
    {
	for (i = 0; i < NODECARD; i++)
	{
	    if (n->branch[i].son >= 0)
		if ((n->level==1) && IsOverFlow(&n->branch[i].minrect)) {
		    if (SOverlap(r, &n->branch[i].minrect))  {
			over = GetOverFlowPage( idxp, n->branch[i].son );
			if ((de=DeleteOFRect(r, over, &newnode, 
					n->branch[i].rect)) == 0) 
			    deleted = FALSE;
			else if (de == 1) {
			    n->branch[i].minrect = MinOFNodeCover(over, n->branch[i].rect);
			    PutOverFlowPage( idxp, over );
			    deleted = TRUE;
			}
			else {
			    n->branch[i].minrect = MinNodeCover( newnode );
			    PutOnePage( idxp, n->branch[i].son, newnode );
			    deleted = TRUE;
			}
			myfreeOFN( over );
		    }
		    else SetOverFlow(&n->branch[i].minrect);
		}
		else if ( SOverlap(r, &(n->branch[i].minrect)) ) {
		    nnn = GetOnePage( idxp, n->branch[i].son );
		    if (!DeleteRect2(idxp, r, nnn))
		    {
			n->branch[i].minrect = MinNodeCover(nnn);
			/*
			 **  Non-leaf nodes not deleted in this version
			 **  of R+- trees
			 */
			PutOnePage( idxp, n->branch[i].son, nnn );
			deleted = TRUE;
		    }
		    myfree( nnn );
		}
	}
	if (deleted)
	    return 0;
	else
	    return 1;
    }
    else  /* a leaf node */
    {
	for (i = 0; i < NODECARD; i++)
	{
	    if ((n->branch[i].son >= 0)
		    && Equal2Rects( n->branch[i].rect, *r )) {
		DisconBranch(n, i);
		EntryCount--;
		return 0;
	    }
	}
	return 1;
    }
}

/*
 ** Delete rect in a list of overflow pages. If the specified rectangle is there,
 ** and delete the rectangle, o/w return 0. After deleting the rectangle, If the
 ** number of the remaining rectangles in the overflow pages is less than 
 ** NODECARD, the overflow page will be changed to normal page.
 **
 */
//( r, over, node, rect )
int DeleteOFRect( struct Rect *r, struct OverFlowNode *over, struct Node **node, struct Rect rect)
{
    register struct OverFlowNode *rp, *rq;
    struct Rect IntersectRect();
    char *myalloc();
    register int i;

#ifdef DEBUG
    extern int magicNum;
    rp = over;
    while( rp!=NULL) {
	for (i=0; i<rp->count; i++)
	    if (rp->rect[i].boundary[0] == magicNum)
		break;
	rp = rp->next2;
    }
#endif

    rp = over;
    while (rp!=NULL) {
	for (i=0; i<rp->count; i++)
	    if (Equal2Rects( *r, rp->rect[i] ))
		break;
	if (i==rp->count) 
	    rp = rp->next2;
	else break;
    }
    if (rp==NULL) return 0;
    rq = over;
    while (rq->next2 != NULL) {
	if (rq->next2->count == 1) {
	    rp->rect[i] = rq->next2->rect[0];
	    rq->next2 = NULL;
	    return 1;
	}
	rq = rq->next2;
    }
    rq->count --;
    rp->rect[i] = rq->rect[rq->count];
    if (over->count == NODECARD) {
	*node = (struct Node *)myalloc( sizeof( struct Node ) );
	for (i=0; i<NODECARD; i++) {
	    (*node)->branch[i].rect = rp->rect[i];
	    (*node)->branch[i].minrect = IntersectRect( &rp->rect[i], &rect );
	    (*node)->branch[i].son = 0;
	}
	(*node)->count = NODECARD;
	(*node)->rect = rect;
	(*node)->pageNo = over->pageNo;
	(*node)->level = 0;
	return 2; 
    }
    else return 1;
}
/* Add a node to the reinsertion list.  All its branches will later
 ** be reinserted into the index structure.
 */
ReInsert(n, ee)
    register struct Node *n;
    register struct ListNode **ee;
{
    register struct ListNode *l;
    struct ListNode *NewListNode();

    l = NewListNode();
    l->node = n;
    l->next = *ee;
    *ee = l;
}

/* Allocate space for a node in the list used in DeletRect to
 ** store Nodes that are too empty.
 */
    struct ListNode *
NewListNode()
{
    char *myalloc();
    return (struct ListNode *) myalloc(sizeof(struct ListNode));
}

FreeListNode(p)
    register struct ListNode *p;
{
    myfree(p);
}

/* Print out the data in a node.
*/
TPrintNode(n)
    struct Node *n;
{
    int i;
    assert(n);

    printf("node");
    if (n->level == 0)
	printf(" LEAF");
    else if (n->level > 0)
	printf(" NONLEAF");
    else
	printf(" TYPE=?");
    printf("  level=%d  count=%d  page=%d\n", n->level, n->count, n->pageNo);

    PrintRectIdent(&n->rect);
    for (i=0; i<n->count; i++)
    {
	printf("branch %d\n", i);
	PrintBranch(&n->branch[i]);
    }

}

/**********************************************************
  relations for intermediate and terminal nodes 
 **********************************************************/

Irel(idxp, r1, r2, rel, mode)
    int idxp;
    register struct Rect *r1;
    register struct Rect *r2;
    char 	*rel;           /* topological or direction relation */
    char 	*mode;          /* mode of relation: OBJ / MBR */
{
    int mbr;

    if (strcmp(mode,"OBJ")==0)
	mbr = 0;
    else if (strcmp(mode,"MBR")==0)
	mbr = 1;
    else
    {
	fprintf(stderr, "ERROR:\taccepted modes are: OBJ MBR\n");
	CloseIndex(idxp, Root);
	exit(1);
    }

    if (strcmp(rel,"OV")==0)
	return MyOverlap(r1, r2, NUMDIMS);
    else if (strcmp(rel,"DJ")==0)
    {
	if (mbr==0)
	    return Disjoint1(r1, r2, NUMDIMS);
	else
	    return MyOverlap(r1, r2, NUMDIMS);
    }
    else if (strcmp(rel,"EO")==0)
    {
	if (mbr==0)
	    return EOverlap1(r1, r2, NUMDIMS);
	else
	    return MyOverlap(r1, r2, NUMDIMS);
    }
    else if (strcmp(rel,"CV")==0)
    {
	if (mbr==0)
	    return Cover1(r1, r2, NUMDIMS);
	else
	    return MyOverlap(r1, r2, NUMDIMS);
    }
    else if (strcmp(rel,"CN")==0)
    {
	if (mbr==0)
	    return Contain1(r1, r2, NUMDIMS);
	else
	    return MyOverlap(r1, r2, NUMDIMS);
    }
    else if (strcmp(rel,"EQ")==0)
    {
	if (mbr==0)
	    return Equal1(r1, r2, NUMDIMS);
	else
	    return MyOverlap(r1, r2, NUMDIMS);
    }
    else if (strcmp(rel,"IN")==0)
    {
	if (mbr==0)
	    return Inside1(r1, r2, NUMDIMS);
	else
	    return MyOverlap(r1, r2, NUMDIMS);
    }
    else if (strcmp(rel,"CB")==0)
    {
	if (mbr==0)
	    return Covered_by1(r1, r2, NUMDIMS);
	else
	    return MyOverlap(r1, r2, NUMDIMS);
    }
    else if (strcmp(rel,"MT")==0)
    {
	if (mbr==0)
	    return Meet1(r1, r2, NUMDIMS);
	else
	    return MyOverlap(r1, r2, NUMDIMS);
    }
    else if (strcmp(rel,"SN")==0)
	return Strong_North1(r1, r2, NUMDIMS);
    else if (strcmp(rel,"WN")==0)
	return Weak_North1(r1, r2, NUMDIMS);
    else if (strcmp(rel,"SBN")==0)
	return Strong_Bounded_North1(r1, r2, NUMDIMS);
    else if (strcmp(rel,"WBN")==0)
	return Weak_Bounded_North1(r1, r2, NUMDIMS);
    else if (strcmp(rel,"SNE")==0)
	return Strong_NorthEast1(r1, r2, NUMDIMS);
    else if (strcmp(rel,"WNE")==0)
	return Weak_NorthEast1(r1, r2, NUMDIMS);
    else if (strcmp(rel,"SL")==0)
	return Same_Level1(r1, r2, NUMDIMS);
    else if (strcmp(rel,"SSL")==0)
	return Strong_Same_Level1(r1, r2, NUMDIMS);
    else if (strcmp(rel,"JN")==0)
	return Just_North1(r1, r2, NUMDIMS);
    else if (strcmp(rel,"NS")==0)
	return North_South1(r1, r2, NUMDIMS);
    else if (strcmp(rel,"N")==0)
	return North1(r1, r2, NUMDIMS);
    else
    {
	fprintf(stderr, "ERROR:\taccepted topological relations are: OV DJ EO CV CN EQ IN CB MT\n");
	fprintf(stderr, "\taccepted direction relations are: SN WN SBN WBN SNE WNE SL SSL JN NS N\n");
	CloseIndex(idxp, Root);
	exit(1);
    }
}

Trel(idxp, r1, r2, rel, mode)
    int idxp;
    register struct Rect *r1;
    register struct Rect *r2;
    char 	*rel;           /* topological or direction relation */
    char 	*mode;          /* mode of relation: OBJ / MBR */
{
    int mbr;

    if (strcmp(mode,"OBJ")==0)
	mbr = 0;
    else if (strcmp(mode,"MBR")==0)
	mbr = 1;
    else
    {
	fprintf(stderr, "ERROR:\taccepted modes are: OBJ MBR\n");
	CloseIndex(idxp, Root);
	exit(1);
    }


    if (strcmp(rel,"OV")==0)
	return MyOverlap(r1, r2, NUMDIMS);
    else if (strcmp(rel,"DJ")==0)
    {
	if (mbr==0)
	    return Disjoint2(r1, r2, NUMDIMS);
	else
	    return Disjoint(r1, r2, NUMDIMS);
    }
    else if (strcmp(rel,"EO")==0)
    {
	if (mbr==0)
	    return EOverlap2(r1, r2, NUMDIMS);
	else
	    return EOverlap(r1, r2, NUMDIMS);
    }
    else if (strcmp(rel,"CV")==0)
    {
	if (mbr==0)
	    return Cover2(r1, r2, NUMDIMS);
	else
	    return Cover(r1, r2, NUMDIMS);
    }
    else if (strcmp(rel,"CN")==0)
    {
	if (mbr==0)
	    return Contain2(r1, r2, NUMDIMS);
	else
	    return Contain(r1, r2, NUMDIMS);
    }
    else if (strcmp(rel,"EQ")==0)
    {
	if (mbr==0)
	    return Equal2(r1, r2, NUMDIMS);
	else
	    return Equal(r1, r2, NUMDIMS);
    }
    else if (strcmp(rel,"IN")==0)
    {
	if (mbr==0)
	    return Inside2(r1, r2, NUMDIMS);
	else
	    return Inside(r1, r2, NUMDIMS);
    }
    else if (strcmp(rel,"CB")==0)
    {
	if (mbr==0)
	    return Covered_by2(r1, r2, NUMDIMS);
	else
	    return Covered_by(r1, r2, NUMDIMS);
    }
    else if (strcmp(rel,"MT")==0)
    {
	if (mbr==0)
	    return Meet2(r1, r2, NUMDIMS);
	else
	    return Meet(r1, r2, NUMDIMS);
    }
    else if (strcmp(rel,"SN")==0)
	return Strong_North(r1, r2, NUMDIMS);
    else if (strcmp(rel,"WN")==0)
	return Weak_North(r1, r2, NUMDIMS);
    else if (strcmp(rel,"SBN")==0)
	return Strong_Bounded_North(r1, r2, NUMDIMS);
    else if (strcmp(rel,"WBN")==0)
	return Weak_Bounded_North(r1, r2, NUMDIMS);
    else if (strcmp(rel,"SNE")==0)
	return Strong_NorthEast(r1, r2, NUMDIMS);
    else if (strcmp(rel,"WNE")==0)
	return Weak_NorthEast(r1, r2, NUMDIMS);
    else if (strcmp(rel,"SL")==0)
	return Same_Level(r1, r2, NUMDIMS);
    else if (strcmp(rel,"SSL")==0)
	return Strong_Same_Level(r1, r2, NUMDIMS);
    else if (strcmp(rel,"JN")==0)
	return Just_North(r1, r2, NUMDIMS);
    else if (strcmp(rel,"NS")==0)
	return North_South(r1, r2, NUMDIMS);
    else if (strcmp(rel,"N")==0)
	return North(r1, r2, NUMDIMS);
    else
    {
	fprintf(stderr, "ERROR:\taccepted topological relations are: OV DJ EO CV CN EQ IN CB MT\n");
	fprintf(stderr, "\taccepted direction relations are: SN WN SBN WBN SNE WNE SL SSL JN NS N\n");
	CloseIndex(idxp, Root);
	exit(1);
    }
}
