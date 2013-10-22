#include <stdio.h>
#include "macros.h"
#include "options.h"
#include "index.h"
#include "global.h"
#include "assert.h"
#include "rectangl.h"

extern struct Rect CoverAll;
long int seed;

int     NO_REFIN;

/*-----------------------------------------------------------------------------
  | Initialize a rectangle to have all 0 coordinates.
  -----------------------------------------------------------------------------*/
void InitRect(struct Rect *r)
{
    int i;
    for (i=0; i<NUMSIDES; i++)
	r->boundary[i] = 0;
}

/*-----------------------------------------------------------------------------
  | Return a rect whose first low side is higher than its opposite side -
  | interpreted as an undefined rect.
  -----------------------------------------------------------------------------*/
NullRect(r)
    struct Rect *r;
{
    int i;

    r->boundary[0] = 1;
    r->boundary[NUMDIMS] = -1;
    for (i=1; i<NUMDIMS; i++) {
	r->boundary[i] = 0;
	r->boundary[i+NUMDIMS] = 0;
    }
}

/*-----------------------------------------------------------------------------
  | Fills in random coordinates in a rectangle.
  | The low side is guaranteed to be less than the high side.
  -----------------------------------------------------------------------------*/
RandomRect(r)
    struct Rect *r;
{
    int i, width;
    for (i = 0; i < NUMDIMS; i++)
    {
	/* width from 1 to 1000 / 4, more small ones */
	width = rand() % (1000 / 4) + 1;

	/* sprinkle a given size evenly but so they stay in [0,100] */
	r->boundary[i] = rand() % (1000-width); /* low side */
	r->boundary[i + NUMDIMS] = r->boundary[i] + width;  /* high side */
    }
}

/*-----------------------------------------------------------------------------
  | Fill in the boundaries for a random search rectangle.
  | Pass in a pointer to a rect that contains all the data,
  | and a pointer to the rect to be filled in.
  | Generated rect is centered randomly anywhere in the data area,
  | and has size from 0 to the size of the data area in each dimension,
  | i.e. search rect can stick out beyond data area.
  -----------------------------------------------------------------------------*/
SearchRect(search, data)
    struct Rect *search, *data;
{
    int i, j, size, center;

    assert(search);
    assert(data);

    for (i=0; i<NUMDIMS; i++)
    {
	j = i + NUMDIMS; /* index for high side boundary */
	if (data->boundary[i] > MININT && data->boundary[j] < MAXINT)
	{
	    size = (rand() % (data->boundary[j] - data->boundary[i] + 1)) / 2;
	    center = data->boundary[i]
		+ rand() % (data->boundary[j] - data->boundary[i] + 1);
	    search->boundary[i] = center - size/2;
	    search->boundary[j] = center + size/2;
	}
	else /* some open boundary, search entire dimension */
	{
	    search->boundary[i] = MININT;
	    search->boundary[j] = MAXINT;
	}
    }
}

/*-----------------------------------------------------------------------------
  | Print out the data for a rectangle.
  -----------------------------------------------------------------------------*/
PrintRect(r)
    struct Rect *r;
{
    int i, j;
    struct Rect new;
    assert(r);

    printf("rect:");
    for (i = 0; i < NUMDIMS; i++)
	printf("\t%d\t%d\n", r->boundary[i], r->boundary[i + NUMDIMS]);
}


/*-----------------------------------------------------------------------------
  | Print out the data for a rectangle.
  -----------------------------------------------------------------------------*/
PrintRectIdent(r)
    struct Rect *r;
{
    int i, j;
    struct Rect new;
    assert(r);

    printf("\trect:");
    printf("\t%d\t%d\n", r->boundary[0], r->boundary[0 + NUMDIMS]);
    for (i = 1; i < NUMDIMS; i++)
	printf("\t\t%d\t%d\n", r->boundary[i], r->boundary[i + NUMDIMS]);
}


/*-----------------------------------------------------------------------------
  | Another version that always prints, no graphics.
  -----------------------------------------------------------------------------*/
PrintRect2(r)
    struct Rect *r;
{
    int i;
    assert(r);
    printf("rect:");
    for (i = 0; i < NUMDIMS; i++)
	printf("\t%d\t%d\n", r->boundary[i], r->boundary[i + NUMDIMS]);
}

/*-----------------------------------------------------------------------------
  | Calculate the n-dimensional area of a rectangle
  -----------------------------------------------------------------------------*/
    double
RectArea(r)
    struct Rect *r;
{
    int i;
    double area;
    assert(r);

    if (Undefined(r))
	return 0;

    area = 1;
    for (i=0; i<NUMDIMS; i++)
	area *= (r->boundary[i+NUMDIMS] - r->boundary[i] + 1);
    return area;
}

/*-----------------------------------------------------------------------------
  | Combine two rectangles, make one that includes both.
  -----------------------------------------------------------------------------*/
struct Rect CombineRect(struct Rect *r, struct Rect *rr)
{
    int i, j;
    struct Rect new;
    assert(r && rr);

    /*------------------------------------------------------------
      If a branch of a non-leaf node has (0, 0, 0, 0) as its minrect,
      It means it has empty decedents. And any rectangle combining 
      this special rectangle (0, 0, 0, 0) will return itself as a
      new minrect of the branch.
      ------------------------------------------------------------*/
    if ((rr->boundary[0]==0) && (rr->boundary[1]==0) &&
	    (rr->boundary[2]==0) && (rr->boundary[3]==0))
	return *r;

    if (Undefined(r))
	return *rr;

    if (Undefined(rr))
	return *r;

    for (i = 0; i < NUMDIMS; i++)
    {
	new.boundary[i] = min(r->boundary[i], rr->boundary[i]);
	j = i + NUMDIMS;
	new.boundary[j] = max(r->boundary[j], rr->boundary[j]);
    }
    return new;
}

/*-----------------------------------------------------------------------------
  | Decide whether two rectangles overlap.
  -----------------------------------------------------------------------------*/
Overlap(r, s)
    struct Rect *r, *s;
{
    int i, j;
    assert(r && s);

    for (i=0; i<NUMDIMS; i++)
    {
	j = i + NUMDIMS;  /* index for high sides */
	if (r->boundary[i] > s->boundary[j] || s->boundary[i] > r->boundary[j])
	    return FALSE;
    }
    return TRUE;
}
SOverlap(r, s)
    struct Rect *r, *s;
{
    int i, j;
    assert(r && s);

    for (i=0; i<NUMDIMS; i++)
    {
	j = i + NUMDIMS;  /* index for high sides */
	if (r->boundary[i] > s->boundary[j] || s->boundary[i] > r->boundary[j])
	    return FALSE;
    }
    return TRUE;
}

/*-----------------------------------------------------------------------------
  | return a rectangle that is the intersection of two rectangles.
  | return the undefined rectangle if they don't overlap or one of them
  | is undefined.
  -----------------------------------------------------------------------------*/
struct Rect IntersectRect(struct Rect *r, struct Rect *rr)
{
    int i, j;
    struct Rect new, nullr;
    assert(r && rr);

    if (Undefined(r) || Undefined(rr) || !Overlap (r,rr)) {
	NullRect(&nullr);
	return (nullr);
    }

    for (i = 0; i < NUMDIMS; i++)
    {
	new.boundary[i] = max(r->boundary[i], rr->boundary[i]);
	j = i + NUMDIMS;
	new.boundary[j] = min(r->boundary[j], rr->boundary[j]);
    }
    return new;
}

/*-----------------------------------------------------------------------------
  | Decide whether rectangle r is contained in rectangle s.
  -----------------------------------------------------------------------------*/
Contained(r, s)
    struct Rect *r, *s;
{
    int i, j, result;
    assert((int)r && (int)s);

    /* undefined rect is contained in any other */
    if (Undefined(r))
	return TRUE;

    /* no rect (except an undefined one) is contained in an undef rect */
    if (Undefined(s))
	return FALSE;

    result = TRUE;
    for (i = 0; i < NUMDIMS; i++)
    {
	j = i + NUMDIMS;  /* index for high sides */
	result = result
	    && r->boundary[i] >= s->boundary[i]
	    && r->boundary[j] <= s->boundary[j];
    }
    return result;
}

/******************************************************************************
 ***********************   TOPOLOGICAL RELATIONS *******************************
 *******************************************************************************/

/*-----------------------------------------------------------------------------
  | Disjoint(r, s, ndims)
  |       Decide whether rect s and rect r are disjoint.
  -----------------------------------------------------------------------------*/
Disjoint(r, s, ndims)
    struct Rect    *r, *s;
    int                     ndims;
{
    if (MyOverlap(r, s, ndims))
	return (FALSE);
    else
	return (TRUE);
}

Disjoint2(r, s, ndims)
    struct Rect    *r, *s;
    int                     ndims;
{
    if (RYX(r,s,ndims,4,6) || RYX(r,s,ndims,4,7) || RYX(r,s,ndims,4,9) || 
	    RYX(r,s,ndims,4,10) || RYX(r,s,ndims,5,6) || RYX(r,s,ndims,5,7) || 
	    RYX(r,s,ndims,5,9) || RYX(r,s,ndims,5,10) || RYX(r,s,ndims,7,6) || 
	    RYX(r,s,ndims,7,7) || RYX(r,s,ndims,7,9) || RYX(r,s,ndims,7,10) || 
	    RYX(r,s,ndims,10,6) || RYX(r,s,ndims,10,7) || RYX(r,s,ndims,10,9) || 
	    RYX(r,s,ndims,10,10) || RYX(r,s,ndims,6,4) || RYX(r,s,ndims,6,5) || 
	    RYX(r,s,ndims,6,7) || RYX(r,s,ndims,6,8) || RYX(r,s,ndims,7,4) || 
	    RYX(r,s,ndims,7,5) || RYX(r,s,ndims,7,7) || RYX(r,s,ndims,7,8) || 
	    RYX(r,s,ndims,9,4) || RYX(r,s,ndims,9,5) || RYX(r,s,ndims,9,7) || 
	    RYX(r,s,ndims,9,8) || RYX(r,s,ndims,10,4) || RYX(r,s,ndims,10,5) || 
	    RYX(r,s,ndims,10,7) || RYX(r,s,ndims,10,8))
	return (FALSE);
    else
    {
	if (RY(r,s,ndims,1) || RY(r,s,ndims,13) ||
		RX(r,s,ndims,1) || RX(r,s,ndims,13))
	    NO_REFIN ++;
	return (TRUE);
    }
}

Disjoint1(r, s, ndims)
    struct Rect    *r, *s;
    int                     ndims;
{
    return (TRUE);
}

/*-----------------------------------------------------------------------------
  | EOverlap(r, s, ndims)
  |	Decide whether rect s strictly overlaps rect r. (Egenhofer)
  -----------------------------------------------------------------------------*/
EOverlap(r, s, ndims)
    struct	Rect	*r, *s;
    int			ndims;
{
    if (MyOverlap(r, s, ndims))
    {
	if (Cover(r, s, ndims) || Contain(r, s, ndims) || 
		Equal(r, s, ndims) || Inside(r, s, ndims) || 
		Covered_by(r, s, ndims) || Meet(r, s, ndims))
	    return (FALSE);
	return (TRUE);
    }
    return (FALSE);
}

EOverlap2(r, s, ndims)
    struct	Rect	*r, *s;
    int			ndims;
{
    if (RX(r,s,ndims,1) || RX(r,s,ndims,2) || 
	    RX(r,s,ndims,12) || RX(r,s,ndims,13) || 
	    RY(r,s,ndims,1) || RY(r,s,ndims,2) || 
	    RY(r,s,ndims,12) || RY(r,s,ndims,13))
	return (FALSE);
    else
    {
	if (RYX(r,s,ndims,4,9) || RYX(r,s,ndims,5,6) || 
		RYX(r,s,ndims,5,9) || RYX(r,s,ndims,5,10) || 
		RYX(r,s,ndims,6,5) || RYX(r,s,ndims,8,9) || 
		RYX(r,s,ndims,9,4) || RYX(r,s,ndims,9,5) || 
		RYX(r,s,ndims,9,8) || RYX(r,s,ndims,10,5))
	    NO_REFIN ++;
	return (TRUE);
    }
}

EOverlap1(r, s, ndims)
    struct	Rect	*r, *s;
    int			ndims;
{
    if (Disjoint(r, s, ndims) || Meet(r, s, ndims))
	return (FALSE);
    else
	return (TRUE);
}

MyOverlap(r, s, ndims)
    struct Rect *r, *s;
    int			ndims;
{
    int i, j;
    assert(r && s);

    for (i=0; i<ndims; i++)
    {
	j = i + ndims;  /* index for high sides */
	if (r->boundary[i] > s->boundary[j] || s->boundary[i] > r->boundary[j])
	    return FALSE;
    }
    return TRUE;
}

/*-----------------------------------------------------------------------------
  | Cover(r, s, ndims)
  |	Decide whether rect s covers rect r
  -----------------------------------------------------------------------------*/
Cover(r, s, ndims)
    struct	Rect	*r, *s;
    int			ndims;
{
    int i, j;
    assert(r && s);

    for (i=0; i<ndims; i++)
    {
	j = i + ndims;  /* index for high sides */
	if (s->boundary[i] > r->boundary[i] || s->boundary[j] < r->boundary[j])
	    return (FALSE);
    }
    if (Contain(r, s, ndims) || Equal(r, s, ndims))
	return (FALSE);
    return (TRUE);
}

Cover2(r, s, ndims)
    struct	Rect	*r, *s;
    int			ndims;
{
    if (RYX(r,s,ndims,4,4) || RYX(r,s,ndims,4,5) || RYX(r,s,ndims,4,7) || 
	    RYX(r,s,ndims,4,8) || RYX(r,s,ndims,5,4) || RYX(r,s,ndims,5,5) || 
	    RYX(r,s,ndims,5,7) || RYX(r,s,ndims,5,8) || RYX(r,s,ndims,7,4) || 
	    RYX(r,s,ndims,7,5) || RYX(r,s,ndims,7,7) || RYX(r,s,ndims,7,8) || 
	    RYX(r,s,ndims,8,4) || RYX(r,s,ndims,8,5) || RYX(r,s,ndims,8,7) || 
	    RYX(r,s,ndims,8,8))
	return (TRUE);
    else
	return (FALSE);
}

Cover1(r, s, ndims)
    struct	Rect	*r, *s;
    int			ndims;
{
    if (Cover(r, s, ndims) || Contain(r, s, ndims) || Equal(r, s, ndims))
	return (TRUE);
    else
	return (FALSE);
}

/*-----------------------------------------------------------------------------
  | Contain(r, s, ndims)
  |	Decide whether rect s contains rect r
  -----------------------------------------------------------------------------*/
Contain(r, s, ndims)
    struct	Rect	*r, *s;
    int			ndims;
{
    int i, j;
    assert(r && s);

    for (i=0; i<ndims; i++)
    {
	j = i + ndims;  /* index for high sides */
	if (s->boundary[i] >= r->boundary[i] || s->boundary[j] <= r->boundary[j])
	    return (FALSE);
    }
    return (TRUE);
}

Contain2(r, s, ndims)
    struct	Rect	*r, *s;
    int			ndims;
{
    if (RYX(r,s,ndims,5,5))
	return (TRUE);
    else
	return (FALSE);
}

Contain1(r, s, ndims)
    struct	Rect	*r, *s;
    int			ndims;
{
    if (Contain(r, s, ndims))
	return (TRUE);
    else
	return (FALSE);
}

/*-----------------------------------------------------------------------------
  | Equal(r, s, ndims)
  |	Decide whether rect s equals rect r
  -----------------------------------------------------------------------------*/
Equal(r, s, ndims)
    struct	Rect	*r, *s;
    int			ndims;
{
    int i, j;
    assert(r && s);

    for (i=0; i<ndims; i++)
    {
	j = i + ndims;  /* index for high sides */
	if (s->boundary[i] != r->boundary[i] || s->boundary[j] != r->boundary[j])
	    return (FALSE);
    }
    return (TRUE);
}

Equal2(r, s, ndims)
    struct	Rect	*r, *s;
    int			ndims;
{
    if (RYX(r,s,ndims,7,7))
	return (TRUE);
    else
	return (FALSE);
}

Equal1(r, s, ndims)
    struct Rect    *r, *s;
    int                     ndims;
{
    if (Cover(r, s, ndims) || Contain(r, s, ndims) || Equal(r, s, ndims))
	return (TRUE);
    else
	return (FALSE);
}

/*-----------------------------------------------------------------------------
  | Inside(r, s, ndims)
  |	Decide whether rect s is inside rect r
  -----------------------------------------------------------------------------*/
Inside(r, s, ndims)
    struct	Rect	*r, *s;
    int			ndims;
{
    return (Contain(s, r, ndims));
}

Inside2(r, s, ndims)
    struct	Rect	*r, *s;
    int			ndims;
{
    if (RYX(r,s,ndims,9,9))
	return (TRUE);
    else
	return (FALSE);
}

Inside1(r, s, ndims)
    struct	Rect	*r, *s;
    int			ndims;
{
    if (Disjoint(r, s, ndims) || Meet(r, s, ndims))
	return (FALSE);
    else
	return (TRUE);
}

/*-----------------------------------------------------------------------------
  | Covered_by(r, s, ndims)
  |	Decide whether rect s is covered_by rect r
  -----------------------------------------------------------------------------*/
Covered_by(r, s, ndims)
    struct	Rect	*r, *s;
    int			ndims;
{
    return (Cover(s, r, ndims));
}

Covered_by2(r, s, ndims)
    struct	Rect	*r, *s;
    int			ndims;
{
    if (RYX(r,s,ndims,6,6) || RYX(r,s,ndims,6,7) || RYX(r,s,ndims,6,9) || 
	    RYX(r,s,ndims,6,10) || RYX(r,s,ndims,7,6) || RYX(r,s,ndims,7,7) || 
	    RYX(r,s,ndims,7,9) || RYX(r,s,ndims,7,10) || RYX(r,s,ndims,9,6) || 
	    RYX(r,s,ndims,9,7) || RYX(r,s,ndims,9,9) || RYX(r,s,ndims,9,10) || 
	    RYX(r,s,ndims,10,6) || RYX(r,s,ndims,10,7) || RYX(r,s,ndims,10,9) || 
	    RYX(r,s,ndims,10,10))
	return (TRUE);
    else
	return (FALSE);
}

Covered_by1(r, s, ndims)
    struct	Rect	*r, *s;
    int			ndims;
{
    if (Disjoint(r, s, ndims) || Meet(r, s, ndims))
	return (FALSE);
    else
	return (TRUE);
}

/*-----------------------------------------------------------------------------
  | Meet(r, s, ndims)
  |	Decide whether rect s meets rect r
  -----------------------------------------------------------------------------*/
Meet(r, s, ndims)
    struct	Rect	*r, *s;
    int			ndims;
{
    int i, j;
    assert(r && s);

    if (MyOverlap(r, s, ndims))
	for (i=0; i<ndims; i++)
	{
	    j = i + ndims;  /* index for high sides */
	    if (s->boundary[i] == r->boundary[j] || 
		    r->boundary[i] == s->boundary[j])
		return (TRUE);
	}
    return (FALSE);
}

Meet2(r, s, ndims)
    struct	Rect	*r, *s;
    int			ndims;
{
    if (RX(r,s,ndims,1) || RX(r,s,ndims,13) || 
	    RY(r,s,ndims,1) || RY(r,s,ndims,13) || 
	    RYX(r,s,ndims,4,9) || RYX(r,s,ndims,5,6) || RYX(r,s,ndims,5,7) || 
	    RYX(r,s,ndims,5,9) || RYX(r,s,ndims,5,10) || RYX(r,s,ndims,6,5) || 
	    RYX(r,s,ndims,7,5) || RYX(r,s,ndims,7,9) || RYX(r,s,ndims,8,9) || 
	    RYX(r,s,ndims,9,4) || RYX(r,s,ndims,9,5) || RYX(r,s,ndims,9,7) || 
	    RYX(r,s,ndims,9,8) || RYX(r,s,ndims,10,5))
	return (FALSE);
    else
	return (TRUE);
}

Meet1(r, s, ndims)
    struct	Rect	*r, *s;
    int			ndims;
{
    if (Disjoint(r, s, ndims))
	return (FALSE);
    else
	return (TRUE);
}

/*-----------------------------------------------------------------------------
  | RYX(r, s, ndims, y, x), RY(r, s, ndims, y), RX(r, s, ndims, x)
  |       Primitive topological functions . x=1..13, y=1..13
  -----------------------------------------------------------------------------*/
RYX(r, s, ndims, y, x)
    struct Rect    *r, *s;
    int                     ndims, y, x;
{
    if (RY(r, s, ndims, y) && RX(r, s, ndims, x))
	return (TRUE);
    else
	return (FALSE);
}

RY(r, s, ndims, y)
    struct Rect    *r, *s;
    int                     ndims, y;
{
    assert(r && s);

    if (y==1)
	return (s->boundary[1] > r->boundary[3]);
    if (y==2)
	return (s->boundary[1] == r->boundary[3]);
    if (y==3)
	return ((s->boundary[3] > r->boundary[3]) &&
		(s->boundary[1] > r->boundary[1]) &&
		(s->boundary[1] < r->boundary[3]));
    if (y==4)
	return ((s->boundary[3] > r->boundary[3]) &&
		(s->boundary[1] == r->boundary[1]));
    if (y==5)
	return ((s->boundary[3] > r->boundary[3]) &&
		(s->boundary[1] < r->boundary[1]));
    if (y==6)
	return ((s->boundary[3] == r->boundary[3]) &&
		(s->boundary[1] > r->boundary[1]));
    if (y==7)
	return ((s->boundary[3] == r->boundary[3]) &&
		(s->boundary[1] == r->boundary[1]));
    if (y==8)
	return ((s->boundary[3] == r->boundary[3]) &&
		(s->boundary[1] < r->boundary[1]));
    if (y==9)
	return ((s->boundary[3] < r->boundary[3]) &&
		(s->boundary[1] > r->boundary[1]));
    if (y==10)
	return ((s->boundary[3] < r->boundary[3]) &&
		(s->boundary[1] == r->boundary[1]));
    if (y==11)
	return ((s->boundary[3] < r->boundary[3]) &&
		(s->boundary[3] > r->boundary[1]) &&
		(s->boundary[1] < r->boundary[1]));
    if (y==12)
	return (s->boundary[3] == r->boundary[1]);
    if (y==13)
	return (s->boundary[3] < r->boundary[1]);
}

RX(r, s, ndims, x)
    struct Rect    *r, *s;
    int                     ndims, x;
{
    assert(r && s);

    if (x==1)
	return (s->boundary[0] > r->boundary[2]);
    if (x==2)
	return (s->boundary[0] == r->boundary[2]);
    if (x==3)
	return ((s->boundary[2] > r->boundary[2]) &&
		(s->boundary[0] > r->boundary[0]) &&
		(s->boundary[0] < r->boundary[2]));
    if (x==4)
	return ((s->boundary[2] > r->boundary[2]) &&
		(s->boundary[0] == r->boundary[0]));
    if (x==5)
	return ((s->boundary[2] > r->boundary[2]) &&
		(s->boundary[0] < r->boundary[0]));
    if (x==6)
	return ((s->boundary[2] == r->boundary[2]) &&
		(s->boundary[0] > r->boundary[0]));
    if (x==7)
	return ((s->boundary[2] == r->boundary[2]) &&
		(s->boundary[0] == r->boundary[0]));
    if (x==8)
	return ((s->boundary[2] == r->boundary[2]) &&
		(s->boundary[0] < r->boundary[0]));
    if (x==9)
	return ((s->boundary[2] < r->boundary[2]) &&
		(s->boundary[0] > r->boundary[0]));
    if (x==10)
	return ((s->boundary[2] < r->boundary[2]) &&
		(s->boundary[0] == r->boundary[0]));
    if (x==11)
	return ((s->boundary[2] < r->boundary[2]) &&
		(s->boundary[2] > r->boundary[0]) &&
		(s->boundary[0] < r->boundary[0]));
    if (x==12)
	return (s->boundary[2] == r->boundary[0]);
    if (x==13)
	return (s->boundary[2] < r->boundary[0]);
}



/******************************************************************************
 ***********************   DIRECTION RELATIONS *********************************
 *******************************************************************************/

/*-----------------------------------------------------------------------------
  | Strong_North(r, s, ndims)
  |       Decide whether rect s is Strong_North of rect r.
  -----------------------------------------------------------------------------*/
Strong_North(r, s, ndims)
    struct Rect    *r, *s;
    int                     ndims;
{
    assert(r && s);
    return (s->boundary[1] > r->boundary[3]);
}

Strong_North1(r, s, ndims)
    struct Rect    *r, *s;
    int                     ndims;
{
    assert(r && s);
    return (s->boundary[3] > r->boundary[3]);
}

/*-----------------------------------------------------------------------------
  | Weak_North(r, s, ndims)
  |       Decide whether rect s is Weak_North of rect r.
  -----------------------------------------------------------------------------*/
Weak_North(r, s, ndims)
    struct Rect    *r, *s;
    int                     ndims;
{
    assert(r && s);
    return ((s->boundary[3] > r->boundary[3]) &&
	    (s->boundary[1] > r->boundary[1]) &&
	    (s->boundary[1] < r->boundary[3]));
}

Weak_North1(r, s, ndims)
    struct Rect    *r, *s;
    int                     ndims;
{
    assert(r && s);
    return ((s->boundary[3] > r->boundary[3]) &&
	    (s->boundary[1] < r->boundary[3]));
}

/*-----------------------------------------------------------------------------
  | Strong_Bounded_North(r, s, ndims)
  |       Decide whether rect s is Strong_Bounded_North of rect r.
  -----------------------------------------------------------------------------*/
Strong_Bounded_North(r, s, ndims)
    struct Rect    *r, *s;
    int                     ndims;
{
    assert(r && s);
    return ((s->boundary[1] > r->boundary[3]) &&
	    (s->boundary[1] > r->boundary[1]) &&
	    (s->boundary[0] > r->boundary[0]) &&
	    (s->boundary[2] < r->boundary[2]) &&
	    (s->boundary[3] > r->boundary[3]));
}

Strong_Bounded_North1(r, s, ndims)
    struct Rect    *r, *s;
    int                     ndims;
{
    assert(r && s);
    return ((s->boundary[3] > r->boundary[3]) &&
	    (s->boundary[0] < r->boundary[2]) &&
	    (s->boundary[2] > r->boundary[0]));
}

/*-----------------------------------------------------------------------------
  | Weak_Bounded_North(r, s, ndims)
  |       Decide whether rect s is Weak_Bounded_North of rect r.
  -----------------------------------------------------------------------------*/
Weak_Bounded_North(r, s, ndims)
    struct Rect    *r, *s;
    int                     ndims;
{
    assert(r && s);
    return ((s->boundary[3] > r->boundary[3]) &&
	    (s->boundary[1] < r->boundary[3]) &&
	    (s->boundary[0] > r->boundary[0]) &&
	    (s->boundary[1] > r->boundary[1]) &&
	    (s->boundary[2] < r->boundary[2]));
}

Weak_Bounded_North1(r, s, ndims)
    struct Rect    *r, *s;
    int                     ndims;
{
    assert(r && s);
    return ((s->boundary[3] > r->boundary[3]) &&
	    (s->boundary[1] < r->boundary[3]) &&
	    (s->boundary[0] < r->boundary[2]) &&
	    (s->boundary[2] > r->boundary[0]));
}

/*-----------------------------------------------------------------------------
  | Strong_NorthEast(r, s, ndims)
  |       Decide whether rect s is Strong_NorthEast of rect r.
  -----------------------------------------------------------------------------*/
Strong_NorthEast(r, s, ndims)
    struct Rect    *r, *s;
    int                     ndims;
{
    assert(r && s);
    return ((s->boundary[1] > r->boundary[3]) &&
	    (s->boundary[0] > r->boundary[2]));
}

Strong_NorthEast1(r, s, ndims)
    struct Rect    *r, *s;
    int                     ndims;
{
    assert(r && s);
    return ((s->boundary[3] > r->boundary[3]) &&
	    (s->boundary[2] > r->boundary[2]));
}

/*-----------------------------------------------------------------------------
  | Weak_NorthEast(r, s, ndims)
  |       Decide whether rect s is Weak_NorthEast of rect r.
  -----------------------------------------------------------------------------*/
Weak_NorthEast(r, s, ndims)
    struct Rect    *r, *s;
    int                     ndims;
{
    assert(r && s);
    return ((s->boundary[3] > r->boundary[3]) &&
	    (s->boundary[2] > r->boundary[2]) &&
	    (s->boundary[1] > r->boundary[1]) &&
	    (s->boundary[0] > r->boundary[0]) &&
	    (s->boundary[1] < r->boundary[3]));
}

Weak_NorthEast1(r, s, ndims)
    struct Rect    *r, *s;
    int                     ndims;
{
    assert(r && s);
    return ((s->boundary[3] > r->boundary[3]) &&
	    (s->boundary[2] > r->boundary[2]) &&
	    (s->boundary[1] < r->boundary[3]));
}

/*-----------------------------------------------------------------------------
  | Same_Level(r, s, ndims)
  |       Decide whether rect s is Same_Level of rect r.
  -----------------------------------------------------------------------------*/
Same_Level(r, s, ndims)
    struct Rect    *r, *s;
    int                     ndims;
{
    assert(r && s);
    return ((s->boundary[1] >= r->boundary[1]) &&
	    (s->boundary[3] <= r->boundary[3]));
}

Same_Level1(r, s, ndims)
    struct Rect    *r, *s;
    int                     ndims;
{
    assert(r && s);
    return ((s->boundary[3] >= r->boundary[1]) &&
	    (s->boundary[1] <= r->boundary[3]));
}

/*-----------------------------------------------------------------------------
  | Strong_Same_Level(r, s, ndims)
  |       Decide whether rect s is Strong_Same_Level of rect r.
  -----------------------------------------------------------------------------*/
Strong_Same_Level(r, s, ndims)
    struct Rect    *r, *s;
    int                     ndims;
{
    assert(r && s);
    return ((s->boundary[1] == r->boundary[1]) &&
	    (s->boundary[3] == r->boundary[3]));
}

Strong_Same_Level1(r, s, ndims)
    struct Rect    *r, *s;
    int                     ndims;
{
    assert(r && s);
    return ((s->boundary[3] >= r->boundary[1]) &&
	    (s->boundary[1] <= r->boundary[3]));
}

/*-----------------------------------------------------------------------------
  | Just_North(r, s, ndims)
  |       Decide whether rect s is Just_North of rect r.
  -----------------------------------------------------------------------------*/
Just_North(r, s, ndims)
    struct Rect    *r, *s;
    int                     ndims;
{
    assert(r && s);
    return (s->boundary[1] == r->boundary[3]);
}

Just_North1(r, s, ndims)
    struct Rect    *r, *s;
    int                     ndims;
{
    assert(r && s);
    return ((s->boundary[3] >= r->boundary[3]) &&
	    (s->boundary[1] <= r->boundary[3]));
}

/*-----------------------------------------------------------------------------
  | North_South(r, s, ndims)
  |       Decide whether rect s is North_South of rect r.
  -----------------------------------------------------------------------------*/
North_South(r, s, ndims)
    struct Rect    *r, *s;
    int                     ndims;
{
    assert(r && s);
    return ((s->boundary[3] > r->boundary[3]) &&
	    (s->boundary[1] < r->boundary[1]));
}

North_South1(r, s, ndims)
    struct Rect    *r, *s;
    int                     ndims;
{
    assert(r && s);
    return ((s->boundary[3] > r->boundary[3]) &&
	    (s->boundary[1] < r->boundary[1]));
}

/*-----------------------------------------------------------------------------
  | North(r, s, ndims)
  |       Decide whether rect s is North of rect r.
  -----------------------------------------------------------------------------*/
North(r, s, ndims)
    struct Rect    *r, *s;
    int                     ndims;
{
    assert(r && s);
    return ((s->boundary[3] > r->boundary[3]) &&
	    (s->boundary[1] > r->boundary[1]));
}

North1(r, s, ndims)
    struct Rect    *r, *s;
    int                     ndims;
{
    assert(r && s);
    return (s->boundary[3] > r->boundary[3]);
}

