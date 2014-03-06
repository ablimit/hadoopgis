#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include "justdoit.h"
#include "cuda/cuda_spatial.h"


void init_spatial_data(spatial_data_t *data)
{
	init_poly_array(&data->polys);
	init_spatial_index(&data->index);
}

void free_spatial_data(spatial_data_t *data)
{
	fini_poly_array(&data->polys);
	fini_spatial_index(&data->index);
	free(data);
}

// Each text line has the following format:
// poly_id, mbr.l mbr.r mbr.b mbr.t, x0 y0, x1 y1, ..., xn yn, x0 y0,
// Note: there is a tailing `,' at the end of each line, to ease parsing
static int load_polys_from_file(
	poly_array_t *polys,
	FILE *file,
	char *parsebuf,
	const int parsebuf_size)
{
	int offset = 0;
	int ipoly = 0;
	char *p, *q;

	// read and parse each text line
	while(fgets(parsebuf, parsebuf_size, file)) {
		if(parsebuf[0] == '\n' || parsebuf[0] == '\0')
			continue;
		polys->offsets[ipoly] = offset;

		// omit prefix chars until the first ','
		p = parsebuf;
		while(*p != ',') p++;
		p++;

		// read in mbr data
		sscanf(p, " %d %d %d %d", &polys->mbrs[ipoly].l, &polys->mbrs[ipoly].r,
			&polys->mbrs[ipoly].b, &polys->mbrs[ipoly].t);

		// omit mbr text
		while(*p != ',') p++;
		p++;

		// parse vertex data
		do {
			q = p;
			while(*q != ',') q++;
			sscanf(p, " %d %d", &polys->x[offset], &polys->y[offset]);
			p = q + 1;
			offset++;
		} while(*p != '\n' && *p != '\0');

		// move on to the next poly
		ipoly++;
	}

	// the last offset indexes beyond the end of x,y arrays
	polys->offsets[ipoly] = offset;
	return 0;
}

int load_polys(poly_array_t *polys, FILE *file)
{
	int readbuf_size = 1024 * 8;
	char *readbuf = NULL;
	int retval = 0;

	readbuf = malloc(readbuf_size);
	if(!readbuf) {
		retval = -1;
		goto out;
	}

	// read number of polygons and total number of vertices
	if(!fgets(readbuf, readbuf_size, file)) {
		retval = -1;
		goto out;
	}
	sscanf(readbuf, "%d, %d\n", &polys->nr_polys, &polys->nr_vertices);

	// to optimize memory layout on cpu/gpu, we allocate a large continuous space
	// that accomodates mbrs, offsets, x and y arrays together; in this manner,
	// only one memory movement is needed to transfer all these data from cpu
	// to gpu
	int size_mbrs = polys->nr_polys * sizeof(mbr_t);
	int size_offsets = (polys->nr_polys + 1) * sizeof(int);
	int size_x = polys->nr_vertices * sizeof(int);
	int size_y = polys->nr_vertices * sizeof(int);

	polys->mbrs = malloc(size_mbrs + size_offsets + size_x + size_y);
	if(!polys->mbrs) {
		retval = -1;
		goto out;
	}
	polys->offsets = (int *)((char *)(polys->mbrs) + size_mbrs);
	polys->x = (int *)((char *)(polys->offsets) + size_offsets);
	polys->y = (int *)((char *)(polys->x) + size_x);

	// load polys data
	if(load_polys_from_file(polys, file, readbuf, readbuf_size)) {
		retval = -1;
		goto out;
	}

out:
	FREE(readbuf);
	return retval;
}

spatial_data_t *load_polys_and_build_index(FILE *file)
{
	spatial_data_t *data = NULL;

	data = malloc(sizeof(spatial_data_t));
	if(!data)
		goto error;
	init_spatial_data(data);

	// load polys
	poly_array_t *polys = &data->polys;
	if(load_polys(polys, file))
		goto error;

	// build index
	spatial_index_t *index = &data->index;
	if(build_spatial_index(index, polys->mbrs, polys->nr_polys, INDEX_R_TREE))
		goto error;

	// bingo!
	goto success;

error:
	if(data) {
		free_spatial_data(data);
		data = NULL;
	}

success:
	return data;
}

float *refine_and_do_spatial_op(
	poly_pair_array_t *poly_pairs,
	poly_array_t *polys1,
	poly_array_t *polys2)
{
	// we do this operation on gpu
	return cuda_clip(
		poly_pairs->nr_poly_pairs, poly_pairs->mbrs,
		poly_pairs->idx1, poly_pairs->idx2,
		polys1->nr_polys + 1, polys1->offsets,
		polys2->nr_polys + 1, polys2->offsets,
		polys1->nr_vertices, polys1->x, polys1->y,
		polys2->nr_vertices, polys2->x, polys2->y);
}

float *just_do_it(
	FILE *file1, FILE *file2,
//	int **idx1, int **idx2,
	float **ratios, int *count)
{
	spatial_data_t *data1 = NULL, *data2 = NULL;
	poly_pair_array_t *poly_pairs = NULL;
	float *result_ratios = NULL;
	int nr_pairs = 0;
	struct timeval t1, t2;

	gettimeofday(&t1, NULL);
	// load polys and build indexes
	data1 = load_polys_and_build_index(file1);
	if(!data1)
		goto out;
	data2 = load_polys_and_build_index(file2);
	if(!data2)
		goto out;
	gettimeofday(&t2, NULL);
	printf("Time on loading and building indexes: %lf s\n", DIFF_TIME(t1, t2));

	gettimeofday(&t1, NULL);
	// filering
	poly_pairs = spatial_filter(&data1->index, &data2->index);
	if(!poly_pairs)
		goto out;
	gettimeofday(&t2, NULL);
	printf("Time on filtering: %lf s\n", DIFF_TIME(t1, t2));

	gettimeofday(&t1, NULL);
	// refinement and spatial operations
	result_ratios = refine_and_do_spatial_op(poly_pairs, &data1->polys, &data2->polys);
	if(result_ratios) {
//		int *temp1 = malloc(poly_pairs->nr_poly_pairs * sizeof(int));
//		if(!temp1) {
//			perror("malloc for idx1 failed");
//			free(result_ratios);
//			result_ratios = NULL;
//			goto out;
//		}
//		int *temp2 = malloc(poly_pairs->nr_poly_pairs * sizeof(int));
//		if(!temp2) {
//			perror("malloc for idx2 failed");
//			free(temp1);
//			free(result_ratios);
//			result_ratios = NULL;
//			goto out;
//		}
//		memcpy(temp1, poly_pairs->idx1, poly_pairs->nr_poly_pairs * sizeof(int));
//		memcpy(temp2, poly_pairs->idx2, poly_pairs->nr_poly_pairs * sizeof(int));
//
//		*idx1 = temp1;
//		*idx2 = temp2;
		*ratios = result_ratios;
		*count = poly_pairs->nr_poly_pairs;
	}
	gettimeofday(&t2, NULL);
	printf("Time on refinement and spatial op: %lf s\n", DIFF_TIME(t1, t2));

out:
	// free stuff
	if(poly_pairs)
		free_poly_pair_array(poly_pairs);
	if(data1)
		free_spatial_data(data1);
	if(data2)
		free_spatial_data(data2);

	return result_ratios;
}

int main(int argc, char *argv[])
{
	float *ratios = NULL;
	int i;

  return 0;
}
