#ifndef GPU_SPATIAL_H
#define GPU_SPATIAL_H

#include "../spatial.h"

#define CUDA_FREE(m)	if(m) cudaFree(m)

#ifdef __cplusplus
//#ifdef IN_GPU_SPATIAL_LIB
extern "C" {
//#endif
#endif

float *clip(
	int stream_no,
	int nr_poly_pairs, const mbr_t *mbrs,
	const int *idx1, const int *idx2,
	int no1, const int *offsets1,
	int no2, const int *offsets2,
	int nv1, const int *x1, const int *y1,
	int nv2, const int *x2, const int *y2);

poly_array_t *gpu_parse(int dno, char *file_name);

int get_gpu_device_count(void);
void init_device_streams(int);
void fini_device_streams();

#ifdef __cplusplus
//#ifdef IN_GPU_SPATIAL_LIB
}
//#endif
#endif

#endif
