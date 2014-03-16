#include "../spatial.h"

extern "C" {
float *cpu_clip(
	int nr_poly_pairs, mbr_t *mbrs,
	int *idx1,int *idx2,
	int no1, int *offsets1,
	int no2, int *offsets2,
	int nv1, int *x1, int *y1,
	int nv2, int *x2, int *y2);

}
