#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include "buffers.h"
#include "constants.h"
#include "spatialindex.h"

dequeue<poly_arrays_buffer_item> buf_poly_arrays(poly_arrays_buffer_size);
extern dequeue<spatial_data_buffer_item> buf_spatial_data;

void *thread_builder(void *param)
{
	poly_arrays_buffer_item poly_arrays;
	spatial_data_buffer_item spatial_data;
	spatial_index_t *index1, *index2;
	struct timeval t1, t2;
	double tot_time = 0.0;
	long n = 0;
	int state, ipair = 0;

	while(true) {
		state = buf_poly_arrays.pull_task(&poly_arrays);

		if(state == 0) {
			gettimeofday(&t1, NULL);

			// build index for poly array 1
			index1 = (spatial_index_t *)malloc(sizeof(spatial_index_t));
			init_spatial_index(index1);
			if(build_spatial_index(index1, poly_arrays.polys_1->mbrs,
				poly_arrays.polys_1->nr_polys, INDEX_HILBERT_R_TREE)) {
			}

			// build index for poly array 2
			index2 = (spatial_index_t *)malloc(sizeof(spatial_index_t));
			init_spatial_index(index2);
			if(build_spatial_index(index2, poly_arrays.polys_2->mbrs,
				poly_arrays.polys_2->nr_polys, INDEX_HILBERT_R_TREE)) {
			}

			gettimeofday(&t2, NULL);
			tot_time += DIFF_TIME(t1, t2);
			n++;

			spatial_data.polys_1 = poly_arrays.polys_1;
			spatial_data.index_1 = index1;
			spatial_data.polys_2 = poly_arrays.polys_2;
			spatial_data.index_2 = index2;

			buf_spatial_data.push_task(&spatial_data);
		}

		else {
			buf_spatial_data.signal_exit();
			break;
		}
	}

	printf("[builder] average time per building: %lf s\n", n > 0 ? tot_time / n : 0.0);

	return NULL;
}
