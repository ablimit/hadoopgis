#ifndef HILBERT_H
#define HILBERT_H

#include "rtree.h"

#define HBT_RESOLUTION 16

typedef struct mbr_index{
	int idx;
	mbr_t *mbr;
	unsigned int hbt_val;
} mbr_idx_t;

extern mbr_t get_mbr(r_tree_node_t *node);

int build_spatial_index_hilbert(
	r_tree_t **rtree,
	mbr_t *mbrs,
	int nr_polys);

void free_spatial_index_hilbert(r_tree_t *rtree);
poly_pair_array_t *spatial_filter_hilbert(
	r_tree_t *rtree1,
	r_tree_t *rtree2);

#endif
