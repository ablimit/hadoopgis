#include <stdio.h>
#include "macros.h"
#include "options.h"
#include "index.h"
#include "global.h"
#include "assert.h"
#include "node.h"

int OFPGCount = 0;

extern int 	Splited;
extern int	pageNo;

// functions 
struct Rect CombineRect(struct Rect *r, struct Rect *rr);
struct Rect IntersectRect(struct Rect *r, struct Rect *rr);
struct OverFlowNode * GetOverFlowPage (int idxp, off_t offset);

int SplitNode (int idxp,
	struct Node *n,     /* node to be split */
	struct Branch *b,
	struct Node **nn,   /* new node */
	struct NodeCut *c)
{
    int coversplitarea, branchbufnum, level, newarea, i, j=0;
    struct Branch newbranch, branchbuf[NODECARD+1];
    struct Rect rect, coversplit;
    //struct Rect ;
    struct Rect dummy1, dummy2;
    struct Node *nnn, *nnnn;
    struct NodeCut cc;
    struct OverFlowNode *overFlowNode, *overflow, *newov;
    int sp;
    int myLevel, myPageNo;
    char s[100];

    assert(n && nn && c);
    myLevel = n->level;
    myPageNo = n->pageNo;

    Splited = TRUE;
#   ifdef PRINT
    printf("Splitting:\n");
    PrintNode(n);
    if (b)
    {
	printf("branch:\n");
	PrintBranch(b);
    }
#   endif

    if (StatFlag)
    {
	if (Deleting)
	    DeSplitCount++;
	else
	    InSplitCount++;
    }

    /* load all the branches into a buffer, initialize old node */
    level = n->level;
    rect = n->rect;
    /* load the branch buffer */



    for (i=0; i<NODECARD; i++)
    {
	/**********************************NOTICE*****************************
	  If its father is splited, it will be splited too. But it may not 
	  be full. So the following statement is not needed.
	  assert(n->branch[i].son+1);    
	 ************************************ho********************************/
	if (n->branch[i].son >= 0)
	    branchbuf[j++] = n->branch[i];
    }
    branchbufnum = n->count;
    if (b)
    {
	branchbuf[branchbufnum] = *b;
	branchbufnum++;
    }
    /* calculate minrect containing all in the set */
    coversplit = branchbuf[0].minrect;
    for (i=1; i<branchbufnum; i++)
	coversplit = CombineRect(&coversplit, &branchbuf[i].minrect);
    coversplitarea = RectArea(&coversplit);
    InitNode(n);

    if (c->axis == ' ')
    {
	assert(branchbufnum == NODECARD+1);
	Partition (branchbuf, (NODECARD+1)/2, &cc, c);
	if (! (c->axis == 'x' || c->axis == 'y')) {
	    overFlowNode = (struct OverFlowNode *)myalloc(sizeof(struct OverFlowNode));
	    InitOFNode( overFlowNode );
	    overFlowNode->count = NODECARD+1;
	    for (i=0; i<NODECARD+1; i++) 
		overFlowNode->rect[i] = branchbuf[i].rect;

	    overFlowNode->pageNo = myPageNo;
	    PutOverFlowPage( idxp, overFlowNode);
	    OFPGCount ++;
	    /****** for trace only ****/
	    overflow = GetOverFlowPage( idxp, myPageNo );
	    return;

	}

#       ifdef PRINT
	printf("Partition - axis: %c  cut: %d\n", c->axis, c->cut);
	printf("Other Partition - axis: %c  cut: %d\n", cc.axis, cc.cut);
#       endif
    }

    *nn = NewNode();
    (*nn)->level = n->level = level;
    (*nn)->rect = n->rect = rect;
    if (c->axis == 'x')  /* x-cut: vertical cut */
    {
	n->rect.boundary[2] = c->cut;
	(*nn)->rect.boundary[0] = c->cut;
    }
    else  /* y-cut: horizontal cut */
    {
	n->rect.boundary[3] = c->cut;
	(*nn)->rect.boundary[1] = c->cut;
    }

    for (i=0; i<branchbufnum; i++)
    {
	if (Overlap (&branchbuf[i].rect, &n->rect))
	{
	    if (Overlap (&branchbuf[i].rect, &(*nn)->rect))
	    {
		if (level == 0)
		{   /* this rect overlaps both, but it is a data rectangle,
		       put it into both nodes without splitting */
		    assert(n->count<NODECARD);
		    branchbuf[i].minrect = IntersectRect(&branchbuf[i].rect, &n->rect);
		    AddBranch (idxp, &branchbuf[i], n, NULL);
		    assert((*nn)->count<NODECARD);
		    branchbuf[i].minrect =
			IntersectRect(&branchbuf[i].rect, &(*nn)->rect);
		    AddBranch (idxp, &branchbuf[i], *nn, NULL);
		}
		else
		{   /* this rect overlaps both, and it is not a data rectangle;
		       put it into both nodes after splitting it */
		    if ((level ==1) && (IsOverFlow( &(branchbuf[i].minrect)))) {
			overflow = GetOverFlowPage( idxp, branchbuf[i].son );
			if ((sp=SplitOFNode(idxp,overflow, &newov,&branchbuf[i].rect,
					&(newbranch.rect), &nnnn, c))==0) {
			    PutOverFlowPage( idxp, overflow );
			    branchbuf[i].minrect = MinOFNodeCover(overflow, branchbuf[i].rect);
			    assert( overflow->pageNo == branchbuf[i].son);
			    AddBranch(idxp, &branchbuf[i], n, NULL);

			    newbranch.son = pageNo;
			    PutOverFlowPage( idxp, newov );
			    OFPGCount ++;
			    newbranch.minrect = MinOFNodeCover(newov, newbranch.rect);
			    AddBranch(idxp, &newbranch, *nn , NULL);
			}
			else if (sp == 1 ) {
			    PutOverFlowPage( idxp, overflow );
			    branchbuf[i].minrect = MinOFNodeCover(overflow, branchbuf[i].rect);
			    assert( overflow->pageNo == branchbuf[i].son);
			    AddBranch(idxp, &branchbuf[i], n, NULL); 

			    newbranch.son = pageNo;
			    PutOnePage( idxp, pageNo, nnnn );
			    if (nnnn->level == 0) pageNo++;
			    else pageNo = pageNo + 2;
			    newbranch.minrect = MinNodeCover(nnnn);
			    assert((*nn)->count<NODECARD);
			    AddBranch (idxp, &newbranch, *nn, NULL);
			}
			else if ( sp == 2 ) {
			    branchbuf[i].son = pageNo;
			    PutOnePage( idxp, pageNo, nnnn );
			    if (nnnn->level == 0) pageNo++;
			    else pageNo = pageNo + 2;
			    branchbuf[i].minrect = MinNodeCover(nnnn);
			    assert((*nn)->count<NODECARD);
			    AddBranch (idxp, &branchbuf[i], n, NULL );

			    PutOverFlowPage( idxp, overflow );
			    newbranch.minrect = MinOFNodeCover(overflow, newbranch.rect);
			    newbranch.son = overflow->pageNo;
			    AddBranch(idxp, &newbranch, *nn, NULL);
			}
			else {
			    printf("SPLITNODE 3 ERROR not that value\n");
			    exit(0);
			}	
		    }
		    else {
			nnn = GetOnePage( idxp, branchbuf[i].son );
			if (! Equal2Rects( branchbuf[i].rect, nnn->rect))
			    printf("SPLITNODE 4 ERROR it is not correct 2\n");
			SplitNode (idxp, nnn, NULL, &nnnn, c);
			PutOnePage( idxp, nnn->pageNo, nnn );
			branchbuf[i].minrect = MinNodeCover(nnn);
			branchbuf[i].rect = nnn->rect;
			assert(n->count<NODECARD);
			AddBranch (idxp, &branchbuf[i], n, NULL);
			myfree( nnn );
			newbranch.son = pageNo;
			PutOnePage( idxp, pageNo, nnnn );
			if (nnnn->level == 0) pageNo++;
			else pageNo = pageNo + 2;
			newbranch.minrect = MinNodeCover(nnnn);
			newbranch.rect = nnnn->rect;
			assert((*nn)->count<NODECARD);
			AddBranch (idxp, &newbranch, *nn, NULL);
			myfree( nnnn );
			if (StatFlag && !Deleting)
			    DownwardSplitCount++;
		    }
		}
	    }
	    else
	    {   /* this rect goes in the old node */
		assert(n->count<NODECARD);
		AddBranch (idxp, &branchbuf[i], n, NULL);
	    }
	}
	else if (Overlap (&branchbuf[i].rect, &(*nn)->rect))
	{
	    /* this rect goes in the new node */
	    assert((*nn)->count<NODECARD);
	    AddBranch (idxp, &branchbuf[i], *nn, NULL);
	}
	else {
	    printf("SPLITNODE 5 ERROR something wrong !\n");
	    exit(1);
	}
    }

    /* record how good the split was for statistics */
    if (StatFlag && !Deleting)
    {
	dummy1 = MinNodeCover(n);
	dummy2 = MinNodeCover(*nn);
	newarea = RectArea(&dummy1) +
	    RectArea(&dummy2);
	SplitMeritSum += (float)coversplitarea / newarea;
    }

#       ifdef PRINT
    printf("group 0:\n");
    PrintNode(n);
    printf("group 1:\n");
    PrintNode(*nn);
    printf("\n");
#       endif

}
/* 
 **
 ** Split overflow node along specified cut line
 **
 */

int SplitOFNode( 
	int idxp,
	struct OverFlowNode *over, struct OverFlowNode **newover,
	struct Rect *r1,
	struct Rect *r2,
	struct Node **newnode,
	struct NodeCut *c)
{
    int i, j=0, k=0;
    struct Branch b;
    struct OverFlowNode *rp, *rq1, *rq2;

#ifdef DEBUG
    extern int magicNum;
    if (over->count<NODECARD)
	printf("SPLITNODE 6 ERROR unbleivbale\n");
    if (over->pageNo == magicNum)
	printf("SPLITNODE 7 ERROR herer here \n");
#endif
    *r2 = *r1;
    if (c->axis == 'x') {
	r1->boundary[NUMDIMS] = c->cut;
	r2->boundary[0] = c->cut;
    }
    else {
	r1->boundary[NUMDIMS+1] = c->cut;
	r2->boundary[1] = c->cut;
    }
    *newnode = (struct Node *)myalloc( sizeof( struct Node ) );
    InitNode( *newnode );
    *newover = (struct OverFlowNode *)myalloc( sizeof( struct OverFlowNode ));
    InitOFNode( *newover );
    rp = over;
    rq1 = over;
    rq2 = (*newover);
    k = 0;
    do {
	for (i=0; i<rp->count; i++) {
	    if (Overlap( &rp->rect[i], r1))
		if ( j != OVERFLOWNODECARD ) 
		    rq1->rect[j++] = rp->rect[i];
		else {
		    rq1->count = j;
		    rq1 = rq1->next2;
		    j = 0;
		    rq1->rect[j++] = rp->rect[i];
		}
	    if (Overlap( &rp->rect[i], r2))
		if ( k != OVERFLOWNODECARD )
		    rq2->rect[k++] = rp->rect[i];
		else {
		    rq2->count = k;
		    k = 0;
		    rq2->next2 = (struct OverFlowNode*)myalloc(sizeof(struct OverFlowNode));
		    rq2 = rq2->next2;
		    InitOFNode( rq2 );
		    rq2->rect[k++] = rp->rect[i];
		}
	}
	rp = rp->next2;
    }
    while (rp !=NULL);
    rq1->count = j;
    rq2->count = k;
    if (j == 0) {
	/**********
	  over->count = (*newover)->count;
	 *************/
	over->count = (*newover)->count;
	myfree( *newover );
	(*newnode)->count = 0;
	(*newnode)->level = 0;
	(*newnode)->rect = *r1;
	return 2;
    }
    if ( k ==0) {
	(*newnode)->count = 0;
	(*newnode)->level = 0;
	(*newnode)->rect = *r2;
	return 1;
    }
    if ( j <= NODECARD ) {
	for (i=0; i<j; i++) {
	    b.rect = over->rect[i];
	    b.minrect = IntersectRect( &(over->rect[i]), r1 );
	    b.son = 0;
	    AddBranch( idxp, &b, *newnode, NULL );
	}
	(*newnode)->count = j;
	(*newnode)->rect = *r1;
	(*newnode)->level = 0;
	rp = (*newover);
	rq1 = over;
	while (rp != NULL) {
	    rp->pageNo = rq1->pageNo;
	    rp = rp->next2;
	    rq1 = rq1->next2;
	}
	*over = **newover;
	return 2;
    }
    if ( k <= NODECARD ) { 
	rq1->next2 = NULL;
	for (i=0; i<k; i++) { 
	    b.rect = (*newover)->rect[i];   
	    b.minrect = IntersectRect( &(*newover)->rect[i], r2 ); 
	    b.son = 0; 
	    AddBranch( idxp, &b, *newnode, NULL ); 
	} 
	(*newnode)->count = k;
	(*newnode)->rect = *r2;
	(*newnode)->level = 0;
	myfree( *newover );
	return 1;
    }
    return 0;
}
