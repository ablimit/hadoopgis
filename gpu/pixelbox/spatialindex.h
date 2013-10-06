#ifndef SPATIAL_INDEX_H
#define SPATIAL_INDEX_H

#include "spatial.h"

// spatial index types
typedef enum
{
	INDEX_UNKNOWN,
	INDEX_R_TREE,
//	INDEX_R_STAR_TREE,
	INDEX_HILBERT_R_TREE,
	INDEX_GRID
} index_scheme_t;

// spatial index
typedef struct spatial_index_struct
{
	index_scheme_t scheme;	// indexing scheme 
	void *index;			// scheme-specific index data
} spatial_index_t;


void init_spatial_index(spatial_index_t *index);
void fini_spatial_index(spatial_index_t *index);
void free_spatial_index(spatial_index_t *index);
int build_spatial_index(
	spatial_index_t *index,
	mbr_t *mbrs,
	int nr_polys,
	index_scheme_t scheme);
poly_pair_array_t *spatial_filter(
	spatial_index_t *index1,
	spatial_index_t *index2);
poly_pair_array_t *spatial_filter_2(
	spatial_index_t *index1,
	poly_array_t *polys2);

#endif
