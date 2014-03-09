#ifndef RSTAR_H
#define RSTAR_H

#include "rtree.h"

int build_spatial_index_r_star(
	r_tree_t **rtree,
	const mbr_t *mbrs,
	const int nr_polys);
void free_spatial_index_r_star(r_tree_t *rtree);
poly_pair_array_t *spatial_filter_r_star(
	const r_tree_t *rtree1,
	const r_tree_t *rtree2);

#endif
