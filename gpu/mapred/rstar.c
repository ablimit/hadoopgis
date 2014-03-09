#include <stdlib.h>
#include "rstar.h"

/*******************************************************************************
 * TODO: R*-Tree routines
 */
int build_spatial_index_r_star(
	r_tree_t **rtree,
	const mbr_t *mbrs,
	const int nr_polys)
{
	return -1;
}

void free_spatial_index_r_star(r_tree_t *rtree)
{
	free(rtree->nodes);
	free(rtree);
}

poly_pair_array_t *spatial_filter_r_star(
	const r_tree_t *rtree1,
	const r_tree_t *rtree2)
{
	return NULL;
}
