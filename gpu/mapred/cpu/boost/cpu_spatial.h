#include "../../spatial.h"

extern "C" {
  float *clip(
	const int nr_poly_pairs, 
  const mbr_t *mbrs,
	const int *idx1, const int *idx2,
	const int no1, const int *offsets1,
	const int no2, const int *offsets2,
	const int nv1, const int *x1, const int *y1,
	const int nv2, const int *x2, const int *y2);

}
