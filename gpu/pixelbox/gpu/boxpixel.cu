#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#define IN_GPU_SPATIAL_LIB
#include "gpu_spatial.h"

//extern cudaStream_t *stream;

#define NR_THREADS_PER_BLOCK	64
#define ROW_PARTITION_FACTOR	3	// how many binary partitions along the row
#define COL_PARTITION_FACTOR	3	// how many binary partitions along the col
#define THRESHOLD_PIXELIZE			(NR_THREADS_PER_BLOCK * NR_THREADS_PER_BLOCK)

#define SIMPLIFY_COMPUTE

extern cudaStream_t *stream;

// decision vector matrices
// rows denote positions of the sampling box relative to the first polygon,
// columns denote positions relative to the second polygon.
// 0 - in, 1 - out, 2 - over
static __device__ const char c_inter[3][3] = {
	{0, 0, 1},
	{0, 0, 0},
	{1, 0, 1}
};

static __device__ const char a_inter[3][3] = {
	{1, 0, 0},
	{0, 0, 0},
	{0, 0, 0}
};

static __device__ const char c_union[3][3] = {
	{0, 0, 0},
	{0, 0, 1},
	{0, 1, 1}
};

static __device__ const char a_union[3][3] = {
	{1, 1, 1},
	{1, 0, 0},
	{1, 0, 0}
};


static inline __device__ int in_poly(
	const int x, const int y,
	const int *xs, const int *ys,
	const int start, const int end)
{
	int nr_edges_crossing = 0;

	for(int i = start; i < end - 1; i++) {
		// whether this edge is parallel to the horizon
		int parallel = (ys[i] == ys[i+1]);

#ifdef SIMPLIFY_COMPUTE

		int intersect = (ys[i+1] > ys[i]) & ((y > ys[i] - 1) & (y < ys[i+1]) & (x < xs[i]));
		intersect |= (ys[i+1] < ys[i]) & ((y > ys[i+1] - 1) & (y < ys[i]) & (x < xs[i+1]));

#else

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
#endif

		// does this edge cross the ray?
		nr_edges_crossing += intersect & (!parallel);
	}

	// an odd number of crossings indicates a pixel laying within the polygon
	return (nr_edges_crossing & 1);
}

// here we assume polygon edges in counter-clockwise order
static inline __device__ int box_position(
	int l, int r, int b, int t,
	const int *x, const int *y,
	const int start, const int end)
{
	int pv_out_box = 1, cross = 0, bc_in_poly = 0;
	int cx = (l + r) / 2, cy = (b + t) / 2;
	int nr_edges_crossing = 0;

	for(int i = start; i < end - 1; i++) {
		// whether this vertex lies outside the box
		pv_out_box &= (x[i] < l + 1) | (x[i] > r - 2) | (y[i] < b + 1) | (y[i] > t - 2);

		// whether there is cross-through
		cross |= (x[i] < l + 1) & (y[i] > b) & (y[i] < t - 1) & (x[i+1] > l);
		cross |= (y[i] < b + 1) & (x[i] > l) & (x[i] < r - 1) & (y[i+1] > b);
		cross |= (x[i] > r - 2) & (y[i] > b) & (y[i] < t - 1) & (x[i+1] < r - 1);
		cross |= (y[i] > t - 2) & (x[i] > l) & (x[i] < r - 1) & (y[i+1] < t - 1);

		// for nc_in_polygon
		int parallel = (y[i] == y[i+1]);
		int intersect = (y[i+1] > y[i]) & ((cy > y[i] - 1) & (cy < y[i+1]) & (cx < x[i]));
		intersect |= (y[i+1] < y[i]) & ((cy > y[i+1] - 1) & (cy < y[i]) & (cx < x[i+1]));
		nr_edges_crossing += intersect & (!parallel);
	}

	bc_in_poly = nr_edges_crossing & 1;

//printf("p: %d, c: %d, b: %d\n", pv_out_box, cross, bc_in_poly);

	return cross * 2 + (1 - cross) * ((1 - pv_out_box) * 2 + pv_out_box * (1 - bc_in_poly));;
}

static inline __device__ void sub_sampbox(
	int l, int w, int b, int h,
	int *new_l, int *new_w, int *new_b, int *new_h)
{
	int h1 = h, w1 = w;
	int nr = 0, nc = 0;
	int ir, ic;

	// calculate row split factor and column split factor
	// since every thread has the same input, there won't be thread divergence
	while(h1 > 1 && nr < ROW_PARTITION_FACTOR) {
		h1 /= 2;
		nr++;
	}
	while(w1 > 1 && nc < COL_PARTITION_FACTOR) {
		w1 /= 2;
		nc++;
	}
	if(nr < ROW_PARTITION_FACTOR)
		nc = ROW_PARTITION_FACTOR + COL_PARTITION_FACTOR - nr;
	else if(nc < COL_PARTITION_FACTOR)
		nr = ROW_PARTITION_FACTOR + COL_PARTITION_FACTOR - nc;

	// do the partitioning
	nr = 1 << nr;
	nc = 1 << nc;
	ir = threadIdx.x / nc;
	ic = threadIdx.x - ir * nc;

	// calculate dimensions of new box
	*new_l = l + ic * (w / nc);
	*new_b = b + ir * (h / nr);
	*new_w = w / nc + (w % nc) * (ic == (nc - 1));
	*new_h = h / nr + (h % nr) * (ir == (nr - 1));
}

// cuda kernel function
static __global__ void kernel_clip(
	const int nr_poly_pairs,
	int *areas_inter, int *areas_union,
	const mbr_t *mbrs, const int *idx1, const int *idx2,
	const int *offsets1, const int *x1, const int *y1,
	const int *offsets2, const int *x2, const int *y2)
{
	// the shared stack: at most 10
	__shared__ char stack_c[NR_THREADS_PER_BLOCK * 10];
	__shared__ int stack_box_l[NR_THREADS_PER_BLOCK * 10];
	__shared__ int stack_box_r[NR_THREADS_PER_BLOCK * 10];
	__shared__ int stack_box_b[NR_THREADS_PER_BLOCK * 10];
	__shared__ int stack_box_t[NR_THREADS_PER_BLOCK * 10];

	for(int i_pair = blockIdx.x; i_pair < nr_poly_pairs; i_pair += gridDim.x) {
		int area_inter = 0, area_union = 0;
		int top = 1;

		if(threadIdx.x == 0) {
			stack_c[0] = 1;
			stack_box_l[0] = mbrs[i_pair].l;
			stack_box_r[0] = mbrs[i_pair].r;
			stack_box_b[0] = mbrs[i_pair].b;
			stack_box_t[0] = mbrs[i_pair].t;
		}

		while(top > 0) {
			top--;
			__syncthreads();

			int l = stack_box_l[top];
			int r = stack_box_r[top];
			int b = stack_box_b[top];
			int t = stack_box_t[top];
			char c = stack_c[top];
//if(threadIdx.x == 0 && c == 1)
//	printf("fetch top: %d, c: %d, l: %d, r: %d, b: %d, t: %d\n", top, c, l, r, b, t);

			if(c == 0)
				continue;

			else {
				int w = r - l;
				int h = t - b;
				int nr_pixels = w * h;

				// apply pixelization method
				if(nr_pixels < THRESHOLD_PIXELIZE) {
//if(threadIdx.x == 0)
//	printf("pixelize top: %d, l: %d, r: %d, b: %d, t: %d\n", top, l, r, b, t);
					// for each pixel assigned to this thread, compute whether it is in
					// poly1 and/or poly2, thus determining whether this pixel is in the
					// intersection and/or union of poly1 and poly2
					for(int i_pixel = threadIdx.x; i_pixel < nr_pixels; i_pixel += blockDim.x) {
						int x = i_pixel % w + l;
						int y = i_pixel / w + b;

						int in1 = in_poly(x, y, x1, y1, offsets1[idx1[i_pair]], offsets1[idx1[i_pair] + 1]);
						int in2 = in_poly(x, y, x2, y2, offsets2[idx2[i_pair]], offsets2[idx2[i_pair] + 1]);

						area_inter += in1 & in2;
						area_union += in1 | in2;
					}
				}

				// continue partitioning sampling boxes
				else {
					sub_sampbox(l, w, b, h, &l, &r, &b, &t);
					r += l;
					t += b;
					int p1 = box_position(l, r, b, t, x1, y1, offsets1[idx1[i_pair]], offsets1[idx1[i_pair] + 1]);
					int p2 = box_position(l, r, b, t, x2, y2, offsets2[idx2[i_pair]], offsets2[idx2[i_pair] + 1]);
					char i = c_inter[p1][p2];
					char u = c_union[p1][p2];
					c = i | u;
					i = a_inter[p1][p2];
					u = a_union[p1][p2];
//if(i | u)
//	printf("sampbox top: %d, %d %d %d %d, p1: %d, p2: %d, i: %d, u: %d\n", top, l, r, b, t, p1, p2, i, u);
					area_inter += (1 - c) * i * (r - l) * (t - b);
					area_union += (1 - c) * u * (r - l) * (t - b);
					stack_box_l[top + 1 + threadIdx.x] = l;
					stack_box_r[top + 1 + threadIdx.x] = r;
					stack_box_b[top + 1 + threadIdx.x] = b;
					stack_box_t[top + 1 + threadIdx.x] = t;
					stack_c[top + 1 + threadIdx.x] = c;
					if(threadIdx.x == 0)
						stack_c[top] = 0;
					top += 1 + blockDim.x;
				}
			}
		} // while top > 1

		areas_inter[blockDim.x * i_pair + threadIdx.x] = area_inter;
		areas_union[blockDim.x * i_pair + threadIdx.x] = area_union;
	} // for each polygon pair
}

float *clip(
	int stream_no,
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
	const int nr_threads_per_block = NR_THREADS_PER_BLOCK;
	int nr_blocks = 0;

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

	cudaSetDevice(stream_no);

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

	nr_blocks = nr_poly_pairs / 10 + 1;
	// launch kernel
	kernel_clip<<<nr_blocks, nr_threads_per_block, 0, stream[stream_no]>>>(
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
//printf("area_inter: %d, area_union: %d\n", area_inter, area_union);
		ratios[i] = (float)area_inter / (float)area_union;
		//printf("overlap: %d, union: %d\n", area_inter, area_union);
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
