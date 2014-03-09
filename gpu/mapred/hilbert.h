#ifndef HILBERT_H
#define HILBERT_H

#include "rtree.h"

int build_spatial_index_hilbert(
	r_tree_t **rtree,
	const mbr_t *mbrs,
	const int nr_polys);
void free_spatial_index_hilbert(r_tree_t *rtree);
poly_pair_array_t *spatial_filter_hilbert(
	const r_tree_t *rtree1,
	const r_tree_t *rtree2);

#endif
