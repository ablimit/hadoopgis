#include <stdlib.h>
#include "spatial.h"


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

void free_poly_array(poly_array_t *polys)
{
	fini_poly_array(polys);
	free(polys);
}

void init_poly_pair_array(poly_pair_array_t *poly_pairs)
{
	poly_pairs->nr_poly_pairs = 0;
	poly_pairs->mbrs = NULL;
	poly_pairs->idx1 = NULL;
	poly_pairs->idx2 = NULL;
}

int make_poly_pair_array(poly_pair_array_t *poly_pairs, int nr_poly_pairs)
{
	int size_mbrs = nr_poly_pairs * sizeof(mbr_t);
	int size_idx = nr_poly_pairs * sizeof(int);

	poly_pairs->mbrs = (mbr_t *)malloc(size_mbrs + 2 * size_idx);
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
