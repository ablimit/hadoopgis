#include <stdlib.h>
#include "spatial.h"

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

void init_poly_array(poly_array_t *polys)
{
	polys->nr_polys = 0;
	polys->mbrs = NULL;
	polys->offsets = NULL;
	polys->x = NULL;
	polys->y = NULL;
}

void fini_poly_array(poly_array_t *polys)
{
	polys->nr_polys = 0;
	FREE(polys->mbrs);
	polys->mbrs = NULL;
	polys->offsets = NULL;
	polys->x = NULL;
	polys->y = NULL;
}

void init_poly_pair_array(poly_pair_array_t *poly_pairs)
{
	poly_pairs->nr_poly_pairs = 0;
	poly_pairs->mbrs = NULL;
	poly_pairs->idx1 = NULL;
	poly_pairs->idx2 = NULL;
}

int make_poly_pair_array(poly_pair_array_t *poly_pairs, const int nr_poly_pairs)
{
	int size_mbrs = nr_poly_pairs * sizeof(mbr_t);
	int size_idx = nr_poly_pairs * sizeof(int);

	poly_pairs->mbrs = (mbr_t*)malloc(size_mbrs + 2 * size_idx);
	if(!poly_pairs->mbrs)
		return -1;
	poly_pairs->idx1 = (int *)((char *)(poly_pairs->mbrs) + size_mbrs);
	poly_pairs->idx2 = (int *)((char *)(poly_pairs->idx1) + size_idx);
	poly_pairs->nr_poly_pairs = nr_poly_pairs;

	return 0;
}

void fini_poly_pair_array(poly_pair_array_t *poly_pairs)
{
	FREE(poly_pairs->mbrs);
}

void free_poly_pair_array(poly_pair_array_t *poly_pairs)
{
	FREE(poly_pairs->mbrs);
	free(poly_pairs);
}
