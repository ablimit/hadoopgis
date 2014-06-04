#ifndef SPATIAL_H
#define SPATIAL_H

#define A_VERY_LARGE_NUM	1073741824
#define A_VERY_SMALL_NUM	(-1073741824)

#define FREE(m)	if(m) free(m)

#define MIN(x, y)	((x) < (y) ? (x) : (y))
#define MAX(x, y)	((x) > (y) ? (x) : (y))

#define DIFF_TIME(t1, t2) ( \
    ((t2).tv_sec + ((double)(t2).tv_usec)/1000000.0) - \
	((t1).tv_sec + ((double)(t1).tv_usec)/1000000.0) \
)


// minimum bounding rectangular
typedef struct mbr_struct
{
	int l, r, b, t;		// left < right, bottom < top
} mbr_t;

// polygons array
typedef struct poly_array_struct
{
	int nr_polys;		// number of polygons in this array
	mbr_t *mbrs;		// mbrs
	int *offsets;		// offset of each poly in x and y arrays
	int nr_vertices;	// number of vertices for all polys in this array
	int *x;			// x coordinates
	int *y;			// y coordinates
} poly_array_t;

// polygon pairs array: indexes into the polygon arrays
// are stored, instead of the actual polygon data
typedef struct poly_pair_array_struct
{
	int nr_poly_pairs;	// number of polygon pairs
	mbr_t *mbrs;		// mbrs of each polygon pair
	int *idx1;			// index to the first of each polygon pair
	int *idx2;			// index to the second of each polygon pair
} poly_pair_array_t;

/*
inline int mbr_area(const mbr_t *mbr);
inline int mbr_area_2(const mbr_t *mbr1, const mbr_t *mbr2);
inline int mbr_area_inc(const mbr_t *mbr, const mbr_t *mbr_inc);
inline int mbr_diff(const mbr_t *mbr1, const mbr_t *mbr2);
inline int in_mbr(const mbr_t *mbr, int x, int y);
inline int mbr_intersect(const mbr_t *mbr1, const mbr_t *mbr2);
inline int mbr_update(mbr_t *mbr, const mbr_t *mbr_inc);
inline void mbr_merge(mbr_t *merge_to, const mbr_t *mbr1, const mbr_t *mbr2);
*/

// area of an mbr
static inline int mbr_area(const mbr_t *mbr)
{
	return (mbr->r - mbr->l) * (mbr->t - mbr->b);
}

// area of the rectangular covering both mbrs
static inline int mbr_area_2(const mbr_t *mbr1, const mbr_t *mbr2)
{
	int l, r, b, t;
	l = MIN(mbr1->l, mbr2->l);
	r = MAX(mbr1->r, mbr2->r);
	b = MIN(mbr1->b, mbr2->b);
	t = MAX(mbr1->t, mbr2->t);
	return (r - l) * (t - b);
}

// enlargement of mbr area in order to combine mbr_inc to mbr
static inline int mbr_area_inc(const mbr_t *mbr, const mbr_t *mbr_inc)
{
	return mbr_area_2(mbr, mbr_inc) - mbr_area(mbr);
}

// whether two mbrs are the same
static inline int mbr_diff(const mbr_t *mbr1, const mbr_t *mbr2)
{
	return !((mbr1->l == mbr2->l) && (mbr1->r == mbr2->r) &&
		(mbr1->b == mbr2->b) && (mbr1->t == mbr2->t));
}

// whether a point falls within a mbr
static inline int in_mbr(const mbr_t *mbr, int x, int y)
{
	return (x >= mbr->l) && (x < mbr->r) && (y <= mbr->t) && (y > mbr->b);
}

// return whether mbr1 intersects with mbr2, providing that the left edge
// belongs to a rectangular while the right edge does not and the top edge
// belongs to a rectangular while the bottom edge does not
static inline int mbr_intersect(const mbr_t *mbr1, const mbr_t *mbr2)
{
	return	(
		(mbr1->l < mbr2->r) && (mbr1->r > mbr2->l) &&
		(mbr1->b < mbr2->t) && (mbr1->t > mbr2->b)
	);
}

// update mbr by combining mbr_inc to it; return whether mbr was actually
// updated
static inline int mbr_update(mbr_t *mbr, const mbr_t *mbr_inc)
{
	int mbr_changed = 0;

	if(mbr->l > mbr_inc->l) {
		mbr->l = mbr_inc->l;
		mbr_changed = 1;
	}
	if(mbr->r < mbr_inc->r) {
		mbr->r = mbr_inc->r;
		mbr_changed = 1;
	}
	if(mbr->b > mbr_inc->b) {
		mbr->b = mbr_inc->b;
		mbr_changed = 1;
	}
	if(mbr->t < mbr_inc->t) {
		mbr->t = mbr_inc->t;
		mbr_changed = 1;
	}

	return mbr_changed;
}

// merge_to is set to the covering rectangular of mbr1 and mbr2
static inline void mbr_merge(mbr_t *merge_to, const mbr_t *mbr1, const mbr_t *mbr2)
{
	merge_to->l = MIN(mbr1->l, mbr2->l);
	merge_to->r = MAX(mbr1->r, mbr2->r);
	merge_to->b = MIN(mbr1->b, mbr2->b);
	merge_to->t = MAX(mbr1->t, mbr2->t);
}

void init_poly_array(poly_array_t *polys);
void fini_poly_array(poly_array_t *polys);
void init_poly_pair_array(poly_pair_array_t *poly_pairs);
int make_poly_pair_array(poly_pair_array_t *poly_pairs, const int nr_poly_pairs);
void fini_poly_pair_array(poly_pair_array_t *poly_pairs);
void free_poly_pair_array(poly_pair_array_t *poly_pairs);

#endif
