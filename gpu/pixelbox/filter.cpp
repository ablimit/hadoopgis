#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include "constants.h"
#include "buffers.h"

dequeue<spatial_data_buffer_item> buf_spatial_data(spatial_data_buffer_size);
extern dequeue<poly_pairs_buffer_item> buf_poly_pairs;

void *thread_filter(void *param)
{
	spatial_data_buffer_item spatial_data;
	poly_pairs_buffer_item poly_pairs;
	struct timeval t1, t2;
	double tot_time = 0.0;
	long n = 0;
	int state, ipair = 0;

	while(true) {
		state = buf_spatial_data.pull_task(&spatial_data);

		if(state == 0) {
			gettimeofday(&t1, NULL);

			// perform spatial filtering
			poly_pairs.poly_pairs = spatial_filter(spatial_data.index_1, spatial_data.index_2);

			gettimeofday(&t2, NULL);
			tot_time += DIFF_TIME(t1, t2);
			n++;

			poly_pairs.polys_1 = spatial_data.polys_1;
			poly_pairs.polys_2 = spatial_data.polys_2;
			free_spatial_index(spatial_data.index_1);
			free_spatial_index(spatial_data.index_2);

			buf_poly_pairs.push_task(&poly_pairs);
		}

		else {
			buf_poly_pairs.signal_exit();
			break;
		}
	}

	printf("[filter] average time per filtering: %lf s\n", n > 0 ? tot_time / n : 0.0);

	return NULL;
}
