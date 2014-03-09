#include <stdio.h>
#include "spatial.h"
#include "spatialindex.h"

// spatial data: polygon array and its spatial index
typedef struct spatial_data_struct
{
	poly_array_t polys;
	spatial_index_t index;
} spatial_data_t;


void init_spatial_data(spatial_data_t *data);
void free_spatial_data(spatial_data_t *data);
int load_polys(poly_array_t *polys, const int did);
spatial_data_t *load_polys_and_build_index(const int did);
float *refine_and_do_spatial_op(poly_pair_array_t *poly_pairs, poly_array_t *polys1,poly_array_t *polys2);
float *crossmatch(float **ratios, int *count);
