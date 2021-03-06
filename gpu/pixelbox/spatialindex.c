#include <stdlib.h>
#include "spatialindex.h"
#include "rtree.h"
#include "hilbert.h"

void init_spatial_index(spatial_index_t *index)
{
	index->scheme = INDEX_UNKNOWN;
	index->index = NULL;
}

void fini_spatial_index(spatial_index_t *index)
{
	switch(index->scheme) {
		case INDEX_R_TREE:
			free_spatial_index_r((r_tree_t *)(index->index));
			break;

//		case INDEX_R_STAR_TREE:
//			free_spatial_index_r_star((r_tree_t *)(index->index));
//			break;

		case INDEX_HILBERT_R_TREE:
			free_spatial_index_hilbert((r_tree_t *)(index->index));
			break;

//		case INDEX_GRID:
//			free_spatial_index_grid((grid_t *)index->index);
//			break;

		default:
			break;
	}

	index->scheme = INDEX_UNKNOWN;
	index->index = NULL;
}

void free_spatial_index(spatial_index_t *index)
{
	fini_spatial_index(index);
	free(index);
}

int build_spatial_index(
	spatial_index_t *index,
	mbr_t *mbrs,
	int nr_polys,
	index_scheme_t scheme)
{
	int ret;

	index->scheme = scheme;
	switch(scheme) {
		case INDEX_R_TREE:
			ret = build_spatial_index_r(
				(r_tree_t **)&index->index, mbrs, nr_polys
			);
			break;

//		case INDEX_R_STAR_TREE:
//			ret = build_spatial_index_r_star(
//				(r_tree_t **)&index->index, mbrs, nr_polys
//			);
//			break;

		case INDEX_HILBERT_R_TREE:
			ret = build_spatial_index_hilbert(
				(r_tree_t **)&index->index, mbrs, nr_polys
			);
			break;

//		case INDEX_GRID:
//			ret = build_spatial_index_grid(
//				(grid_t **)&index->index, mbrs, nr_polys
//			);
//			break;

		default:
			ret = -1;
			break;
	}

	return ret;
}

// filtering based on the indexes built on both poly arrays
poly_pair_array_t *spatial_filter(
	spatial_index_t *index1,
	spatial_index_t *index2)
{
	poly_pair_array_t *poly_pairs;

	if(index1->scheme != index2->scheme)
		return NULL;

	switch(index1->scheme) {
		case INDEX_R_TREE:
			poly_pairs = spatial_filter_r(
				(r_tree_t *)(index1->index), (r_tree_t *)(index2->index)
			);
			break;

//		case INDEX_R_STAR_TREE:
//			poly_pairs = spatial_filter_r_star(
//				(r_tree_t *)(index1->index), (r_tree_t *)(index2->index)
//			);
//			break;

		case INDEX_HILBERT_R_TREE:
			poly_pairs = spatial_filter_hilbert(
				(r_tree_t *)(index1->index), (r_tree_t *)(index2->index)
			);
			break;

//		case INDEX_GRID:
//			poly_pairs = spatial_filter_grid(
//				(grid_t *)(index1->index), (grid_t *)(index2->index)
//			);
//			break;

		default:
			poly_pairs = NULL;
			break;
	}

	return poly_pairs;
}

// filtering based on the index built on one poly array only
poly_pair_array_t *spatial_filter_2(
	spatial_index_t *index1,
	poly_array_t *polys2)
{
	return NULL;
}
