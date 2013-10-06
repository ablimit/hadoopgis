#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>
#include <sched.h>
#include <sys/time.h>
#include "gpu/gpu_spatial.h"
#include "cpu/cpu_spatial.h"
#include "buffers.h"
#include "constants.h"

dequeue<poly_pairs_buffer_item> buf_poly_pairs(poly_pairs_buffer_size);
extern file_names_buffer buf_file_names;

void do_migration();

void *thread_aggregator(void *param)
{
	poly_pairs_buffer_item poly_pairs;
	int gpu_no = *(int *)param;
	struct timeval t1, t2;
	double tot_time = 0.0;
	long n = 0;
	long nr_pairs = 0;
	pthread_t tid;
	float *ratios;
	int state;

	// this is a gpu-resilient thread
	if(gpu_no > 0) {
		gpu_no--;
		if(pthread_create(&tid, NULL, thread_aggregator, (void *)&gpu_no)) {
			fprintf(stderr, "failed to create sub aggregator thread\n");
			exit(1);
		}

		// start working
		while(true) {
			state = buf_poly_pairs.pull_task(&poly_pairs, &buf_file_names.wl_diverter);

			if(state == 0) {
				gettimeofday(&t1, NULL);

				// perform spatial operations: AIP/AUP
				ratios = gpu_clip(
					gpu_no,
					poly_pairs.poly_pairs->nr_poly_pairs, poly_pairs.poly_pairs->mbrs,
					poly_pairs.poly_pairs->idx1, poly_pairs.poly_pairs->idx2,
					poly_pairs.polys_1->nr_polys + 1, poly_pairs.polys_1->offsets,
					poly_pairs.polys_2->nr_polys + 1, poly_pairs.polys_2->offsets,
					poly_pairs.polys_1->nr_vertices, poly_pairs.polys_1->x, poly_pairs.polys_1->y,
					poly_pairs.polys_2->nr_vertices, poly_pairs.polys_2->x, poly_pairs.polys_2->y);

				gettimeofday(&t2, NULL);
				tot_time += DIFF_TIME(t1, t2);
				n++;
				nr_pairs += poly_pairs.poly_pairs->nr_poly_pairs;

				free_poly_pair_array(poly_pairs.poly_pairs);
				free_poly_array(poly_pairs.polys_1);
				free_poly_array(poly_pairs.polys_2);
				FREE(ratios);
			}
			else {
				break;
			}
		}

		printf("[aggregator %d] %ld tasks processed, %ld poly pairs procssed, tot time %lf s, average time per aggregation: %lf s\n", gpu_no, n, nr_pairs, tot_time, n > 0 ? tot_time / n : 0.0);

		pthread_join(tid, NULL);
	}

	// this is the daemon thread for compute adaptation
	else {
		do_migration();
	}

	return NULL;
}

void do_migration()
{
	poly_pairs_buffer_item poly_pairs;
	struct timeval t1, t2;
	double tot_time = 0.0;
	long n = 0;
	float *ratios;
//return;

	while(true) {
		if(!buf_poly_pairs.try_lock()) {

repeat:
			if(buf_poly_pairs.exiting) {
				buf_poly_pairs.unlock();
				break;
			}

			if(buf_poly_pairs.is_over_loading()) {
				buf_poly_pairs.pull_task_nolock(&poly_pairs);
				buf_poly_pairs.unlock();

//				printf("diverting workload from GPU\n");
				gettimeofday(&t1, NULL);

				// process this task on CPU
				ratios = cpu_clip(
					poly_pairs.poly_pairs->nr_poly_pairs, poly_pairs.poly_pairs->mbrs,
					poly_pairs.poly_pairs->idx1, poly_pairs.poly_pairs->idx2,
					poly_pairs.polys_1->nr_polys + 1, poly_pairs.polys_1->offsets,
					poly_pairs.polys_2->nr_polys + 1, poly_pairs.polys_2->offsets,
					poly_pairs.polys_1->nr_vertices, poly_pairs.polys_1->x, poly_pairs.polys_1->y,
					poly_pairs.polys_2->nr_vertices, poly_pairs.polys_2->x, poly_pairs.polys_2->y);

				gettimeofday(&t2, NULL);
				tot_time += DIFF_TIME(t1, t2);
				n++;

				free_poly_pair_array(poly_pairs.poly_pairs);
				free_poly_array(poly_pairs.polys_1);
				free_poly_array(poly_pairs.polys_2);
				FREE(ratios);
			}
			else {
				buf_poly_pairs.wait_for_congestion();
				goto repeat;
			}
		}
		else {
			sched_yield();
		}
	}

	printf("[aggregate-migrator] %d tasks processed, average time per diversion: %lf s\n", n, n > 0 ? tot_time / n : 0.0);
	return;
}
