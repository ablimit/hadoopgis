#include <stdio.h>
#include "macros.h"
#include "options.h"
#include "index.h"
#include "global.h"
#include "assert.h"

/* Make a new node and initialize to have all branch cells empty.
*/
struct Node *
NewNode()
{
        register struct Node *n;
        char *myalloc();

        NodeCount++;
        n = (struct Node *) myalloc (sizeof(struct Node));
        InitNode(n);
        return n;
}

FreeNode(p)
register struct Node *p;
{
        NodeCount--;
        if (p->level == 0)
                LeafCount--;
        else
                NonLeafCount--;
        myfree(p);
}

/* Initialize a Node structure.
*/
InitNode(n)
register struct Node *n;
{
        register int i;
        n->count = 0;
        n->level = -1;
        InitRect(&(n->rect));
        for (i = 0; i < NODECARD; i++)
                InitBranch(&(n->branch[i]));
}

/* Initialize a Node structure.
*/
InitOFNode(n)
register struct OverFlowNode *n;
{
        register int i;
        n->count = 0;
	n->pageNo = UNKNOWN;
	n->next2 = NULL;
        for (i = 0; i < OVERFLOWNODECARD; i++)
            InitRect(&(n->rect[i]));
}

/* Initialize one branch cell in a node.
*/
InitBranch(b)
register struct Branch *b;
{
        InitRect(&(b->minrect));
        InitRect(&(b->rect));
        b->son = -1;
}

/* Print out the data in a node.
*/
PrintNode(n)
struct Node *n;
{
        int i;
        assert(n);

#       ifdef PRINT
                printf("node");
                if (n->level == 0)
                        printf(" LEAF");
                else if (n->level > 0)
                        printf(" NONLEAF");
                else
                        printf(" TYPE=?");
                printf("  level=%d  count=%d  page=%d\n", n->level, n->count, n->pageNo);

		PrintRect(&n->rect);
                for (i=0; i<n->count; i++)
                {
                        printf("branch %d\n", i);
                        PrintBranch(&n->branch[i]);
                }
#       endif PRINT

}

PrintBranch(b)
struct Branch *b;
{
#               ifdef PRINTINDEX
                        printf("  son in page %d\n", b->son);
#               endif

		printf ("min");
                PrintRect(&(b->minrect));
                PrintRect(&(b->rect));
}

/* Find the smallest rectangle that includes all rectangles in
** branches of a node.
*/
struct Rect
NodeCover(n)
register struct Node *n;
{
        register int i, flag;
        struct Rect r, CombineRect();
        assert(n);

        InitRect(&r);
        flag = 1;
        for (i = 0; i < NODECARD; i++)
                if (n->branch[i].son>=0)
                {
                        if (flag)
                        {
                                r = n->branch[i].rect;
                                flag = 0;
                        }
                        else
                                r = CombineRect(&r, &(n->branch[i].rect));
                }
        return r;
}

/* Find the smallest rectangle that includes all of the minimum rectangles in
** branches of a node.
*/
struct Rect
MinNodeCover(n)
register struct Node *n;
{
        register int i, flag;
        struct Rect r, CombineRect();
	register int IsOF = FALSE;
 	void SetOverFlow();

        assert(n);

        InitRect(&r);
        flag = 1;
        for (i = 0; i < NODECARD; i++) {
		if (IsOverFlow( &n->branch[i].minrect)) IsOF = TRUE;
                if (n->branch[i].son>=0)
                {
                        if (flag)
                        {
                                r = n->branch[i].minrect;
                                flag = 0;
                        }
                        else
                                r = CombineRect(&r, &(n->branch[i].minrect));
                }
	        if ( IsOF ) {
		    SetOverFlow( &n->branch[i].minrect );
		    IsOF = FALSE;
		}
	}
        return r;
}

/* Find the smallest rectangle that includes all of the minimum rectangles in
** the overflow node.
*/
struct Rect
MinOFNodeCover(overflow, rect)
register struct OverFlowNode *overflow;
register struct Rect rect;
{
        register int i, flag;
        struct Rect r, CombineRect(), IntersectRect();
	register struct OverFlowNode *rp;
	void SetOverFlow();

        assert(overflow);
        InitRect(&r);
        flag = 1;
	rp = overflow;
	r = overflow->rect[0];
	while ( rp != NULL ) {
            for (i = 0; i < rp->count; i++)
                r = CombineRect(&r, &(rp->rect[i]));
	    rp = rp->next2;
	}
	r = IntersectRect( &r, &rect );
	SetOverFlow( &r );
        return r;
}

/* Pick a branch.  Pick the one that will need the smallest increase
** in area to accomodate the new rectangle.  This will result in the
** least total area for the covering rectangles in the current node.
** In case of a tie, pick the one which was smaller before, to get
** the best resolution when searching.
*/
int
PickBranch(r, n)
register struct Rect *r;
register struct Node *n;
{
        register struct Rect *rr;
        register int i, flag, increase, bestIncr, area, bestArea;
        int best;
        struct Rect CombineRect();
	struct Rect dummy;
        assert(r && n);

        flag = 1;
        for (i=0; i<NODECARD; i++)
        {
                if (n->branch[i].son >= 0)
                {
                        rr = &n->branch[i].rect;
                        area = RectArea(rr);
		    	dummy = CombineRect(r, rr);
                        increase = RectArea(&dummy) - area;
                        if (increase <  bestIncr || flag)
                        {
                                best = i;
                                bestArea = area;
                                bestIncr = increase;
                                flag = 0;
                        }
                        else if (increase == bestIncr && area < bestArea)
                        {
                                best = i;
                                bestArea = area;
                                bestIncr = increase;
                        }
#                       ifdef PRINT
                                printf("i=%d  area before=%d  area after=%d  increase=%d\n",
                                i, area, area+increase, increase);
#                       endif
                }
        }
#       ifdef PRINT
                printf("\tpicked %d\n", best);
#       endif
        return best;
}

/* Add a branch to a node.  Split the node if necessary.
** Returns 0 if node not split.  Old node updated.
** Returns 1 if node split, sets *new to address of new node.
** Old node updated, becomes one of two.
*/
int
AddBranch(idxp, b, n, new)
int idxp;
register struct Branch *b;
register struct Node *n;
register struct Node **new;
{
        register int i;
        struct NodeCut cut;

        assert(b);
        assert(n);

if ((n->level >0) && (b->son ==0))
  printf("not permited\n");

        if (n->count < NODECARD)  /* split won't be necessary */
        {
                for (i = 0; i < NODECARD; i++)  /* find empty branch */
                {
                        if (n->branch[i].son < 0)
                        {
                                n->branch[i] = *b;
                                n->count++;
/* printf("EMPTY BRANCH AT %d NEW COUNT is %d\n",i,n->count);
fflush(stdout); */
                                break;
                        }
                }
                assert(i<=NODECARD);
                return 0;
        }
        else
        {
                if (StatFlag)
                {
                        if (Deleting)
                                DeTouchCount++;
                        else
                                InTouchCount++;
                }
                assert(new);
                cut.axis = ' ';
                SplitNode(idxp, n, b, new, &cut);
		if (! (cut.axis == 'x' || cut.axis == 'y')) /* overflow */
		    return 2;
                if (n->level == 0)
                        LeafCount++;
                else
                        NonLeafCount++;
                return 1;
        }
}

/* Disconnect a dependent node.
*/
DisconBranch(n, i)
register struct Node *n;
register int i;
{
        assert(n && i>=0 && i<NODECARD);
        assert(n->branch[i].son+1);

        InitBranch(&(n->branch[i]));
        n->count--;
}

/* See if two rectangle are the same */
/* added on 4/22/90 */
int
Equal2Rects(r1, r2)
struct Rect r1, r2;
{
    register int i;
    for(i=0; i<NUMDIMS; i++)
        if ((r1.boundary[i] != r2.boundary[i]) ||
            (r1.boundary[i+NUMDIMS] != r2.boundary[i+NUMDIMS]))
                return FALSE;
    return TRUE;
}


/* See if a data rectangle already exists in a leaf node.
*/
int
RectInNode(r, n)         /* modified on 4/22/90 */
struct Rect r;
register struct Node *n;
{
        register int i;

        assert(n);
	assert(&r);
	assert(n->level + 1);

        for (i=0; i<NODECARD; i++)
		if (Equal2Rects(r, n->branch[i].rect)) 
			return TRUE;
        return FALSE;
}

/*
**
**  If the rectangle is in the overflow node, return TRUE, o/w FALSE
**
*/
int
RectInOFN( r, overflow )
register struct Rect r;
register struct OverFlowNode *overflow;
{
    register int i;
    
    for (i=0; i<overflow->count; i++)
	if (Equal2Rects(overflow->rect[i], r))
	    return TRUE;
    return FALSE;
}

/*
**
** If the branch is overflow, return TRUE, O/w FALSE
**
*/
int IsOverFlow( n )
register struct Rect *n;
{
    int tmp;
 
    if (n->boundary[0] > n->boundary[NUMDIMS]) {
        tmp = n->boundary[0];
        n->boundary[0] = n->boundary[NUMDIMS];
        n->boundary[NUMDIMS] = tmp;
        return TRUE;
    }
    else return FALSE;
}
/*
**
** Over flow node will be recognized by switch the two x coordinates
**
*/
void SetOverFlow( r )
register struct Rect *r;
{
    int tmp;

    tmp = r->boundary[0];
    r->boundary[0] = r->boundary[NUMDIMS];
    r->boundary[NUMDIMS] = tmp;
}
