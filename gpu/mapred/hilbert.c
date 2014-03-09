#include <stdlib.h>
#include "hilbert.h"

/*******************************************************************************
 * TODO: hilbert r-tree routines
 */
int build_spatial_index_hilbert(
	r_tree_t **rtree,
	const mbr_t *mbrs,
	const int nr_polys)
{
	return -1;
}

void free_spatial_index_hilbert(r_tree_t *rtree)
{
	free(rtree->nodes);
	free(rtree);
}

poly_pair_array_t *spatial_filter_hilbert(
	const r_tree_t *rtree1,
	const r_tree_t *rtree2)
{
	return NULL;
}
