#include <stdio.h>
#include "macros.h"
#include "options.h"
#include "index.h"
#include "global.h"
#include "assert.h"
#include "partitio.h"

/*--------------------------------------------------------------
| Determine best partition of a node to be split
--------------------------------------------------------------*/

Partition (ba, ff, cc, dd)
struct Branch ba[];  /* branch array */
register int ff;     /* fill factor */
struct NodeCut *cc, *dd;
{
    register int axis;
    register int i, j; 
    int left, right; 
    int cuts, minCuts;
    int dif, minDif;
    double minAreas, area, Areas();
int llll, rrrr;
    
    minAreas = MAXAREA;
    minCuts = MAXINT;
    minDif = MAXINT;
    axis = XAxis; /* axis is x */
    for (i=0; i<= NODECARD; i++) {
	left = 0;
	right = 0;
        for (j = 0; j <= NODECARD; j++) 
	    if (ba[i].rect.boundary[axis] >= ba[j].rect.boundary[axis]) {
		left ++;
		if (ba[i].rect.boundary[axis] <= ba[j].rect.boundary[axis+NUMDIMS])
		    right++;
	    }
	    else right++;
	if ((left<=NODECARD) && (right<=NODECARD)) {
	    area = Areas(ba[i].rect.boundary[axis], ba, axis, &cuts);
	    dif = abs(left-right);
/* printf("FIRST X try\tLEFT=%d\tRIGHT=%d\ti=%d\tArea=%f\tCuts=%d\n",left,right,i,area,cuts); */
	    if (area < minAreas) {
		minAreas = area;
	        cc->axis = 'x';
		cc->cut = ba[i].rect.boundary[axis];
	    }
	    if (cuts < minCuts || (cuts == minCuts && dif < minDif)) {
		llll = left;
		rrrr = right;
		minDif = dif;
		minCuts = cuts;
	        dd->axis = 'x';
		dd->cut = ba[i].rect.boundary[axis];
	    }
	}
	left = 0; 
        right = 0; 
        for (j = 0; j <= NODECARD; j++) 
            if (ba[i].rect.boundary[axis+NUMDIMS] >= ba[j].rect.boundary[axis]) {
                left ++; 
                if (ba[i].rect.boundary[axis+NUMDIMS] <= ba[j].rect.boundary[axis+NUMDIMS]) 
                    right++;
            }
            else right++;
        if ((left<=NODECARD) && (right<=NODECARD)) {
            area = Areas(ba[i].rect.boundary[axis+NUMDIMS], ba, axis, &cuts); 
	    dif = abs(left-right);
/* printf("SECOND X try\tLEFT=%d\tRIGHT=%d\ti=%d\tArea=%f\tCuts=%d\n",left,right,i,area,cuts); */
            if (area < minAreas) {
		minAreas = area;
                cc->axis = 'x';
                cc->cut = ba[i].rect.boundary[axis+NUMDIMS];
            } 
	    if (cuts < minCuts || (cuts == minCuts && dif < minDif)) {
		llll = left;
		rrrr = right;
		minDif = dif;
		minCuts = cuts;
	        dd->axis = 'x';
		dd->cut = ba[i].rect.boundary[axis+NUMDIMS];
	    }
        }
    }
    axis = YAxis; /* axis is y */
    for (i=0; i<= NODECARD; i++) {
        left = 0; 
        right = 0; 
        for (j = 0; j <= NODECARD; j++) 
            if (ba[i].rect.boundary[axis] >= ba[j].rect.boundary[axis]) {
                left ++; 
                if (ba[i].rect.boundary[axis] <= ba[j].rect.boundary[axis+NUMDIMS]) 
                    right++;
            }
            else right++;
        if ((left<=NODECARD) && (right<=NODECARD)) {
            area = Areas(ba[i].rect.boundary[axis], ba, axis, &cuts); 
	    dif = abs(left-right);
/* printf("FIRST Y try\tLEFT=%d\tRIGHT=%d\ti=%d\tArea=%f\tCuts=%d\n",left,right,i,area,cuts); */
            if (area < minAreas) {
		minAreas = area;
                cc->axis = 'y';
                cc->cut = ba[i].rect.boundary[axis];
            } 
	    if (cuts < minCuts || (cuts == minCuts && dif < minDif)) {
		llll = left;
		rrrr = right;
		minDif = dif;
		minCuts = cuts;
	        dd->axis = 'y';
		dd->cut = ba[i].rect.boundary[axis];
	    }
        } 
        left = 0;  
        right = 0;  
        for (j = 0; j <= NODECARD; j++)  
            if (ba[i].rect.boundary[axis+NUMDIMS] >= ba[j].rect.boundary[axis]) {
                left ++;  
                if (ba[i].rect.boundary[axis+NUMDIMS] <= ba[j].rect.boundary[axis+NUMDIMS]) 
                    right++; 
            } 
            else right++; 
        if ((left<=NODECARD) && (right<=NODECARD)) { 
            area = Areas(ba[i].rect.boundary[axis+NUMDIMS], ba, axis, &cuts);  
	    dif = abs(left-right);
/* printf("SECOND Y try\tLEFT=%d\tRIGHT=%d\ti=%d\tArea=%f\tCuts=%d\n",left,right,i,area,cuts); */
            if (area < minAreas) { 
		minAreas = area;
                cc->axis = 'y';  
                cc->cut = ba[i].rect.boundary[axis+NUMDIMS]; 
            }   
	    if (cuts < minCuts || (cuts == minCuts && dif < minDif)) {
		llll = left;
		rrrr = right;
		minDif = dif;
		minCuts = cuts;
	        dd->axis = 'y';
		dd->cut = ba[i].rect.boundary[axis+NUMDIMS];
	    }
        }
    }	
/* printf (" I DECIDE LEFT=%d RIGHT=%d\n",llll,rrrr); */
    return (1);
}
		

double
Areas(cut, ba, axis, cuts)
register cut;
struct Branch ba[];
register int axis;
register int *cuts;
{
    register int i, j;
    struct Rect rect1, rect2, temprect, CombineRect();
    int rect1cnt, rect2cnt;
    double RectArea();

    NullRect(&rect1);
    NullRect(&rect2);
    rect1cnt = rect2cnt = 0;
    *cuts = 0;
    for (i = 0; i < NODECARD+1; i++)
    {
        if (ba[i].rect.boundary[axis+2] <= cut)
        {
            rect1 = CombineRect(&rect1, &ba[i].minrect); rect1cnt++;
        }
        else if (ba[i].rect.boundary[axis] >= cut)
        {
            rect2 = CombineRect(&rect2, &ba[i].minrect); rect2cnt++;
        }
        else
        {   /* this rect overlaps the cut */
            if (ba[i].minrect.boundary[axis+2] <= cut)
                rect1 = CombineRect(&rect1, &ba[i].minrect);
            else if (ba[i].minrect.boundary[axis] >= cut)
                rect2 = CombineRect(&rect2, &ba[i].minrect);
            else
            {   /* the minrect also overlaps the cut */
                temprect = ba[i].minrect;
                temprect.boundary[axis+2] = cut;
                rect1 = CombineRect(&rect1, &temprect);
                temprect = ba[i].minrect;
                temprect.boundary[axis] = cut;
                rect2 = CombineRect(&rect2, &temprect);
            }
            rect1cnt++;
            rect2cnt++;
	    *cuts = *cuts + 1;
        }
    }    
    if (rect1cnt > NODECARD || rect2cnt > NODECARD) 
	return MAXAREA;
    else {
/*     PrintRect(&rect1);
    PrintRect(&rect2); */
    return (RectArea(&rect1) + RectArea(&rect2));
    }
}

