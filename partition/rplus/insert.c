
#include <stdio.h>
#include "macros.h"
#include "options.h"
#include "index.h"
#include "global.h"
#include "assert.h"

extern int 	pageNo;

/* Insert a data rectangle into an index structure.
** InsertRect provides for splitting the root;
** The level argument specifies the number of steps up from the leaf
** level to insert; e.g. a data rectangle goes in at level = 0.
** InsertRect2 does the recursion.
*/
static int InRPtree;
static struct Rect MaxArea = { MININT, MININT, MAXINT, MAXINT };
int Splited;
int
InsertRect(idxp, r, id, root, level)
int idxp;
register struct Rect *r;
register int id;
register struct Node **root;
register int level;
{
        register int i;
        register struct Node *newroot;
        struct Node *newnode, *NewNode();
        struct Branch b;
        struct Rect MinNodeCover(), MinOFNodeCover();
	struct OverFlowNode *overflow, *GetOverFlowPage();
        int AddBranch(), ins;

        assert(r && root);
        assert(level >= 0 && level <= (*root)->level);
	InRPtree = FALSE;
	Splited = FALSE;
        for (i=0; i<NUMDIMS; i++)
                assert(r->boundary[i] <= r->boundary[NUMDIMS+i]);

#       ifdef PRINT
/******************************ho*************
                printf("InsertRect  level=%d\n", level);
                fflush(stdout);
******************************ho*************/
#       endif

        if (StatFlag)
        {
                if (Deleting)
                        ReInsertCount++;
                else
                        InsertCount++;
        }
        if (!Deleting)
                RectCount++;


        if ((ins = InsertRect2(idxp, r, id, *root, &newnode, level))>0)  /* root was split */
        {
                if (StatFlag)
                {
                        if (Deleting)
                                DeTouchCount++;
                        else
                                InTouchCount++;
                }

		if (ins == 1 ) {
printf("I AM GROWING THE TREE TALLER. INS=%d ID=%d\n",ins,id);
                    newroot = NewNode();  /* grow a new root, make tree taller */
                    NonLeafCount++;
      		    PutOnePage( idxp, pageNo, *root );
                    newroot->level = (*root)->level + 1;
                    newroot->rect = MaxArea;
		    for (i=0; i<NUMDIMS; i++) {
            		newroot->rect.boundary[NUMDIMS+i] = MAXINT;
            		newroot->rect.boundary[i] = MININT;
       		    }
		    newroot->pageNo = 0;
                    b.minrect = MinNodeCover(*root);
                    b.son = pageNo;
		    if (newroot->level == 1) pageNo++;
		    else pageNo = pageNo + 2;
                    b.rect = (*root)->rect;
                    AddBranch(idxp, &b, newroot, NULL);
		    PutOnePage( idxp, pageNo, newnode );
                    b.minrect = MinNodeCover(newnode);
                    b.son = pageNo;
		    if (newroot->level == 1) pageNo++;
		    else pageNo = pageNo + 2;
                    b.rect = newnode->rect;
                    AddBranch(idxp, &b, newroot, NULL);
		    myfree( root );
 		    myfree( newnode );
                    *root = newroot;
                    EntryCount += 2;
                }
	        else if (ins == 2 ) {
printf("COULD NOT FIND PARTITION LINE. INS=%d ID=%d\n",ins,id); 
		    InitNode( *root );
		    (*root)->rect = MaxArea;
		    (*root)->level = 1;
		    overflow = GetOverFlowPage( idxp, pageNo -1 );
		    b.minrect = MinOFNodeCover(overflow, (*root)->rect );
		    b.rect = MaxArea;
		    b.son = overflow->pageNo;
		    AddBranch( idxp, &b, *root, NULL );
		    myfree( overflow );
		}
	}
	if (InRPtree) {
	    printf("------> The rectangle is in the R* tree\n");
	    return FALSE;
	}
	else {
/*	printf("I AM DONE INSERTING ID=%d\n",id); */
	return TRUE;}
}

/* Inserts a new data rectangle into the index structure.
** Recursively descends tree, propagates splits back up.
** Returns 0 if node was not split.  Old node updated.
** If node was split, returns 1 and sets the pointer pointed to by
** new to point to the new node.  Old node updated to become one of two.
** The level argument specifies the number of steps up from the leaf
** level to insert; e.g. a data rectangle goes in at level = 0.
*/

/*
**
** This version for R+ -trees does not handle insertions of levels
** other than level 0.
**
*/
int
InsertRect2(idxp, r, id, n, new, level)
int idxp;
register struct Rect *r;
register int id;
register struct Node *n, **new;
register int level;
{
        register int i, done, needsdone;
        int RectInNode();
	int Updates, ins;
	struct OverFlowNode *overflow, *GetOverFlowPage();
        struct Rect MinNodeCover(), CombineRect(), IntersectRect();
	struct Rect dummy;
        struct Branch b;
        struct Node *nnn, *GetOnePage();
        struct Node *n2;
struct Rect rrrr, rr11, rr22;

	if ((r==NULL) || (n==NULL) || (new==NULL))
	printf("INSERT 1 ERROR problem is here\n");
        assert(r && n && new);
        if ((level < 0) || (level > n->level))
printf("INSERT 2 ERROR problem is here now\n");
        assert(level >= 0 && level <= n->level);

        if (StatFlag)
        {
                if (Deleting)
                        DeTouchCount++;
                else
                        InTouchCount++;
        }
/* printf("InsertRect2: I AM INSERTING ID=%d CURRENT-LEV=%d LEVEL=%d\n",id,n->level,level); */

        /* Still above level for insertion, go down tree recursively */
        if (n->level > level)
        {
            done = i = 0;
            needsdone = n->count;
            while (done < needsdone) 
              if ((n->branch[i].son > 0)&&
                  Overlap (r, &n->branch[i].rect)) {
		if ((n->level == 1) && IsOverFlow( &(n->branch[i].minrect))) {
		    overflow = GetOverFlowPage( idxp, n->branch[i].son );
		    if ((ins=InsertRect2OFN( idxp, *r, overflow,&(n->branch[i].rect),
							new))==0) {
			if (!Splited) InRPtree = TRUE;
                	return -1;
		    }
		    else if (ins == 1) {
			PutOverFlowPage( idxp, overflow );
			dummy = CombineRect(r, &(n->branch[i].minrect));
			rr11 = n->branch[i].minrect;
			n->branch[i].minrect = IntersectRect(&n->branch[i].rect,								&dummy);
			rrrr = MinOFNodeCover( overflow, n->branch[i].rect );
			IsOverFlow( &rrrr );
			if (! Equal2Rects( rrrr, n->branch[i].minrect )) {
  				printf("@@@@@@@@@@@@@@@@\n");
  				if (overflow->count <15)
					overflow->count --;
				rr22 = MinOFNodeCover( overflow, n->branch[i].rect );
				if (! Equal2Rects( rr11, rr22 )) {
	  				printf("INSERT 3 ERROR must be a bug\n");
					printf("###############################\n");
     				}
			}
	
			SetOverFlow( &n->branch[i].minrect );
		    }
		    else if (ins == 2) {
			PutOverFlowPage( idxp, overflow );
			n->branch[i].minrect = MinOFNodeCover( overflow, n->branch[i].rect );
			b.minrect = MinNodeCover( *new );
			b.rect = (*new)->rect;
			b.son = pageNo;
			PutOnePage( idxp, pageNo, *new );
			if ( (*new)->level == 0) pageNo++;
			else pageNo = pageNo + 2;
			if (n->count < NODECARD)
			    AddBranch( idxp, &b, n, NULL);
			else {
			    myfree( overflow );
			    return AddBranch( idxp, &b, n, new );
			}
		    }
		    else if (ins == 3 ) {
			printf("INSERT 4 ERROR donothing----<<<<<<<<<<<<<<<<>>>>>>>>>>\n");
			SetOverFlow( &n->branch[i].minrect );
		    }
                    done++; i++; /** not sure it is right */
		    myfree( overflow );
		}
		else {
                    nnn = GetOnePage( idxp, n->branch[i].son );
		    Updates = TRUE;
		    if (! Equal2Rects( n->branch[i].rect, nnn->rect)) {
			printf("INSERT 5 ERROR it is wrong INS=%d ID=%d PAGE=%d\n",ins,id,n->branch[i].son);
			PrintRectIdent(&n->branch[i].rect);
			PrintRectIdent(&nnn->rect);
		    }
                    if ((ins=InsertRect2(idxp, r,id, nnn, &n2, level))==0)
                    {
                        /* son was not split */
		        dummy = CombineRect(r, &(n->branch[i].minrect));
                        n->branch[i].minrect = IntersectRect(&n->branch[i].rect,&dummy);
                        done++; i++;
                    }
		    else if (ins == -1 ) /* rectangle is in son */
		    {
			done++; i++;
			Updates = FALSE;
		    }
		    else if (ins == 2) { /* overflow first happen in the node*/
printf("COULD NOT FIND PARTITION LINE. INS=%d ID=%d\n",ins,id);
			dummy = CombineRect(r, &(n->branch[i].minrect));
                        n->branch[i].minrect = IntersectRect(&n->branch[i].rect,&dummy);
			SetOverFlow( &(n->branch[i].minrect) );
			Updates = FALSE;
			done ++; i++;
		    }
                    else if (ins == 1)  /* son was split */
                    {
                        n->branch[i].rect = nnn->rect;
                        n->branch[i].minrect = MinNodeCover(nnn);
		        PutOnePage( idxp, nnn->pageNo, nnn );
                        b.son = pageNo;
                        b.rect = n2->rect;
                        b.minrect = MinNodeCover(n2);
		        PutOnePage( idxp, pageNo, n2 );
			if (n2->level == 0) pageNo++;
			else pageNo = pageNo + 2;
		        myfree( n2 );
                        EntryCount++;
		        Updates = FALSE;
                        if (n->count < NODECARD)
                        { /* son was split and there is room for it in the node */
/* printf("SON WAS SPLIT BUT THERE IS ROOM level=%d count=%d\n",n->level,n->count); */
                            AddBranch(idxp, &b, n, NULL);
                            needsdone++;
                        }
                        else { /* son was split and no more room in the node */
			    myfree( nnn );
			    assert( n->level > 0 )
/* printf("SON WAS SPLIT BUT THERE IS NO ROOM level=%d count=%d\n",n->level,n->count); */
                            return AddBranch(idxp, &b, n, new);
		        }
                    }
	            if ( Updates ) 
		        PutOnePage( idxp, nnn->pageNo, nnn );
		    myfree( nnn );
		}
              }
              else
              {
                  if (n->branch[i].son) done++;
                  i++;
              }
            return 0;
        }

        /* Have reached level for insertion. Add rect, split if necessary */
        else if (n->level == level)
        {
/*TIMOS
	    if  (!RectInNode(*r, n)) */
            {
		rrrr = n->rect;
                b.rect = *r;
                b.minrect = IntersectRect(r, &n->rect);
                b.son = 5000-id;
/*                b.son = 0; */
                /* son field of leaves contains tid of data record */
                EntryCount++;
                return AddBranch(idxp, &b, n, new);
            }
/* TIMOS
            else {
		if (!Splited) InRPtree = TRUE;
                return -1;
	    } */
        }

        else
        {
            /* Not supposed to happen */
            assert (FALSE);
            return 0;
        }
}

/*
**
**  Insert rectangles to overflow node
**
*/
int InsertRect2OFN( idxp, r, overflow, rect, newnode )
int idxp;
register struct Rect r;
register struct OverFlowNode *overflow;
register struct Rect *rect;
register struct Node **newnode;
{
    register int i, j, k, side, ind, count;
    register int overlap, left=0, right=0, up=0, down=0;
    register struct OverFlowNode *rp, *rq;
    struct Branch b;

printf("I INSERT IN OF NODE\n");
PrintRectIdent(&r);
    rp = overflow;
    count = 0;

    do {
   	count += rp->count;
        for (i=0; i<rp->count; i++) {
	    if (Equal2Rects( rp->rect[i], r )) return 0;
	    if (rp->rect[i].boundary[NUMDIMS] <= r.boundary[0]) 
		    left++;
	    if (rp->rect[i].boundary[NUMDIMS+1] <= r.boundary[1]) 
		    down ++;
	    if (rp->rect[i].boundary[0] >= r.boundary[NUMDIMS]) 
		    right ++;
	    if (rp->rect[i].boundary[1] >= r.boundary[NUMDIMS+1]) 
		    up ++;
        }
	rq = rp;
 	rp = rp->next2;
    } while (rp != NULL);

    side = left; ind = 0;
    if (side < right) {
	side = right;
	ind = NUMDIMS;
    }
    if (side < down) {
	side = down;
	ind = 1;
    }
    if (side < up) {
	side = up;
	ind = NUMDIMS+1;
    }
    
    if (count - side > Thresh) {
 	if (rq->count < OVERFLOWNODECARD) {
	    rq->rect[rq->count] = r;
	    rq->count ++;
	}
	else {
	    rp = (struct OverFlowNode *)myalloc(sizeof(struct OverFlowNode));
	    InitOFNode( rp );
	    rp->count = 1;
	    rp->next2 = NULL;
	    rp->rect[0] = r;
	    rq->next2 = rp;
	}    
        return 1;
    }
    
    
    *newnode=(struct Node *)myalloc(sizeof(struct Node));
    InitNode( *newnode );
    (*newnode)->level = 0;
    (*newnode)->rect = *rect;
    (*newnode)->rect.boundary[ind] = r.boundary[ind];
    if (ind >= NUMDIMS) 
	rect->boundary[ind-NUMDIMS] = r.boundary[ind];
    else rect->boundary[ind+NUMDIMS] = r.boundary[ind];
    k=0;
    rp = overflow;
    rq = overflow;
    do {
        for (i=0; i<rp->count; i++) {
	    if (Overlap(&rp->rect[i], &(*newnode)->rect)) {
	        b.rect = rp->rect[i];
	        b.minrect = IntersectRect( &b.rect, &(*newnode)->rect);
	        b.son = 0;
	        AddBranch( idxp, &b, *newnode, NULL );
	    }
	    if (Overlap(&rp->rect[i], rect)) 
		if ( k < OVERFLOWNODECARD )
	            rq->rect[k++] = rp->rect[i];
		else {
		    rq->count = k;
		    k = 0;
		    rq = rq->next2;
		    rq->rect[k++] = rp->rect[i];
 		}
        }
	rp = rp->next2;
    } while ( rp != NULL );
    rq->count = k;
    b.rect = r;
    b.minrect = IntersectRect( &r, &(*newnode)->rect );
    b.son = 0;
    AddBranch( idxp, &b, *newnode, NULL );
    return 2;

}
