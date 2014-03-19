#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include "../spatial.h"
#define IN_CUDA_SPATIAL_LIB
#include "cuda_spatial.h"


// compute whether point (x, y) lays within a polygon whose contour
// is specified by xs and ys in the range of [istart, iend); the last
// vertex repeats the first vertex
static inline __device__ int in_poly(
	const int x, const int y,
	const int *xs, const int *ys,
	const int start, const int end)
{
	int nr_edges_crossing = 0;

	// for each edge (xs[i], ys[i])->(xs[i+1], ys[i+1]), we see whether it
	// crosses the ray shooting from (x,y) horizontally to infinitely far right
	for(int i = start; i < end - 1; i++) {
		// whether this edge is parallel to the horizon
		int parallel = (ys[i] == ys[i+1]);

		// for an upward edge
		int intersect =
			(ys[i+1] > ys[i]) &
			(
				((y == ys[i]) & ((x < xs[i]) | (x == xs[i]))) |
				(
					(y > ys[i]) & (y < ys[i+1]) &
					(x < xs[i] + (y - ys[i]) *(xs[i+1] - xs[i]) * 1.0 / (ys[i+1] - ys[i] + parallel))
				)
			);

		// for a downward edge
		intersect |=
			(ys[i+1] < ys[i]) &
			(
				((y == ys[i+1]) & ((x < xs[i+1]) | (x == xs[i+1]))) |
				(
					(y > ys[i+1]) & (y < ys[i]) &
					(x < xs[i+1] + (y - ys[i+1]) *(xs[i+1] - xs[i]) * 1.0 / (ys[i+1] - ys[i] + parallel))
				)
			);

		// does this edge cross the ray?
		nr_edges_crossing += intersect & (!parallel);
	}

	// an odd number of crossings indicates a pixel laying within the polygon
	return (nr_edges_crossing & 1);
}

// cuda kernel function
static __global__ void kernel_clip(
	const int nr_poly_pairs,
	int *areas_inter, int *areas_union,
	const mbr_t *mbrs, const int *idx1, const int *idx2,
	const int *offsets1, const int *x1, const int *y1,
	const int *offsets2, const int *x2, const int *y2)
{
	// this thread block processes polygon pairs in the range of
	// [i_pair_start, i_pair_end)
	int nr_pairs_per_block = nr_poly_pairs / gridDim.x + 1;
	int i_pair_start = nr_pairs_per_block * blockIdx.x;
	int i_pair_end = i_pair_start + nr_pairs_per_block;
	if(i_pair_end > nr_poly_pairs) {
		i_pair_end = nr_poly_pairs;
	}

	// for each polygon pair, compute the areas of the intersection and the union.
	// this computation is partitioned among threads in the same block, with each
	// thread computing only some pixels of the mbr
	for(int i_pair = i_pair_start; i_pair < i_pair_end; i_pair++) {
		int mbr_width = mbrs[i_pair].r - mbrs[i_pair].l + 1;
		int mbr_height = mbrs[i_pair].t - mbrs[i_pair].b + 1;
		int nr_pixels_per_thread = mbr_width * mbr_height / blockDim.x + 1;
		int area_inter = 0, area_union = 0;

		// for each pixel assigned to this thread, compute whether it is in
		// poly1 and/or poly2, thus determining whether this pixel is in the
		// intersection and/or union of poly1 and poly2
		for(int i_pixel = 0; i_pixel < nr_pixels_per_thread; i_pixel++) {
			int x = (blockDim.x * i_pixel + threadIdx.x) % mbr_width + mbrs[i_pair].l;
			int y = (blockDim.x * i_pixel + threadIdx.x) / mbr_width + mbrs[i_pair].b;

			int in1 = in_poly(x, y, x1, y1, offsets1[idx1[i_pair]], offsets1[idx1[i_pair] + 1]);
			int in2 = in_poly(x, y, x2, y2, offsets2[idx2[i_pair]], offsets2[idx2[i_pair] + 1]);

			area_inter += in1 & in2;
			area_union += in1 | in2;
		}

		areas_inter[blockDim.x * i_pair + threadIdx.x] = area_inter;
		areas_union[blockDim.x * i_pair + threadIdx.x] = area_union;
	}
}

float *clip(
	const int nr_poly_pairs, const mbr_t *mbrs,	// mbr of each poly pair
	const int *idx1, const int *idx2,			// index to poly_array 1 and 2
	const int no1, const int *offsets1,			// offset to poly_arr1's vertices
	const int no2, const int *offsets2,			// offset to poly_arr2's vertices
	const int nv1, const int *x1, const int *y1,// poly_arr1's vertices
	const int nv2, const int *x2, const int *y2)// poly_arr2's vertices
{
	int size_poly_pairs, size_polys1, size_polys2, size_areas;
	float *ratios = NULL;
	int *areas_inter = NULL, *areas_union = NULL;
	int *gpu_areas_inter = NULL, *gpu_areas_union = NULL;
	mbr_t *gpu_mbrs = NULL;
	int *gpu_idx1 = NULL, *gpu_idx2 = NULL;
	int *gpu_offsets1 = NULL, *gpu_x1 = NULL, *gpu_y1 = NULL;
	int *gpu_offsets2 = NULL, *gpu_x2 = NULL, *gpu_y2 = NULL;
	const int nr_blocks = 256, nr_threads_per_block = 256;

	// allocate cpu memory for result arrays
	ratios = (float *)malloc(nr_poly_pairs * sizeof(float));
	if(!ratios) {
		perror("malloc error for ratios\n");
		goto error;
	}

	size_areas = nr_poly_pairs * nr_threads_per_block * sizeof(int);
	areas_inter = (int *)malloc(size_areas);
	if(!areas_inter) {
		perror("malloc error for areas_inter\n");
		goto error;
	}

	areas_union = (int *)malloc(size_areas);
	if(!areas_union) {
		perror("malloc error for areas_union\n");
		goto error;
	}

	// allocate device memories: due to optimization in the memory layout
	// of poly_pair_array_t and poly_array_t, mbrs, idx1 and idx2 are
	// continuous in memory space; the same for offsets1, x1, y1 and
	// offsets2, x2 and y2.
	if(cudaMalloc((void **)&gpu_areas_inter, size_areas) != cudaSuccess) {
		fprintf(stderr, "cuda malloc error for gpu_areas_inter\n");
		goto error;
	}

	if(cudaMalloc((void **)&gpu_areas_union, size_areas) != cudaSuccess) {
		fprintf(stderr, "cuda malloc error for gpu_areas_union\n");
		goto error;
	}

	size_poly_pairs = nr_poly_pairs * (sizeof(mbr_t) + 2 * sizeof(int));
	size_polys1 = (no1 + 2 * nv1) * sizeof(int);
	size_polys2 = (no2 + 2 * nv2) * sizeof(int);

	if(cudaMalloc((void **)&gpu_mbrs, size_poly_pairs) != cudaSuccess) {
		fprintf(stderr, "cuda malloc error for gpu_mbrs\n");
		goto error;
	}
	gpu_idx1 = (int *)((char *)gpu_mbrs + nr_poly_pairs * sizeof(mbr_t));
	gpu_idx2 = (int *)((char *)gpu_idx1 + nr_poly_pairs * sizeof(int));

	if(cudaMalloc((void**)&gpu_offsets1, size_polys1) != cudaSuccess) {
		fprintf(stderr, "cuda malloc error for gpu_offsets1\n");
		goto error;
	}
	gpu_x1 = (int *)((char *)gpu_offsets1 + no1 * sizeof(int));
	gpu_y1 = (int *)((char *)gpu_x1 + nv1 * sizeof(int));

	if(cudaMalloc((void**)&gpu_offsets2, size_polys2) != cudaSuccess) {
		fprintf(stderr, "cuda malloc error for gpu_offsets2\n");
		goto error;
	}
	gpu_x2 = (int *)((char *)gpu_offsets2 + no2 * sizeof(int));
	gpu_y2 = (int *)((char *)gpu_x2 + nv2 * sizeof(int));

	// transfer data to gpu memory
	if(cudaMemcpy(gpu_mbrs, mbrs, size_poly_pairs, cudaMemcpyHostToDevice) != cudaSuccess) {
		fprintf(stderr, "cuda memcpy error for gpu_mbrs\n");
		goto error;
	}

	if(cudaMemcpy(gpu_offsets1, offsets1, size_polys1, cudaMemcpyHostToDevice) != cudaSuccess) {
		fprintf(stderr, "cuda memcpy error for gpu_offsets1\n");
		goto error;
	}

	if(cudaMemcpy(gpu_offsets2, offsets2, size_polys2, cudaMemcpyHostToDevice) != cudaSuccess) {
		fprintf(stderr, "cuda memcpy error for gpu_offsets2\n");
		goto error;
	}

	// launch kernel
	kernel_clip<<<nr_blocks, nr_threads_per_block>>>(
		nr_poly_pairs,
		gpu_areas_inter, gpu_areas_union,
		gpu_mbrs, gpu_idx1, gpu_idx2,
		gpu_offsets1, gpu_x1, gpu_y1,
		gpu_offsets2, gpu_x2, gpu_y2
	);

	// get results back
	cudaMemcpy(areas_inter, gpu_areas_inter, size_areas, cudaMemcpyDeviceToHost);
	cudaMemcpy(areas_union, gpu_areas_union, size_areas, cudaMemcpyDeviceToHost);

	// compute ratios
	for(int i = 0; i < nr_poly_pairs; i++) {
		int area_inter = 0, area_union = 0;
		for(int j = 0; j < nr_threads_per_block; j++) {
			area_inter += areas_inter[i * nr_threads_per_block + j];
			area_union += areas_union[i * nr_threads_per_block + j];
		}
		ratios[i] = (float)area_inter / (float)area_union;
	}

	goto success;

error:
	FREE(ratios);
	ratios = NULL;

success:
	FREE(areas_inter);
	FREE(areas_union);
	CUDA_FREE(gpu_areas_inter);
	CUDA_FREE(gpu_areas_union);
	CUDA_FREE(gpu_mbrs);
	CUDA_FREE(gpu_offsets1);
	CUDA_FREE(gpu_offsets2);

	return ratios;
}
