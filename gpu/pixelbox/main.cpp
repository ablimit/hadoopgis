#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <sys/time.h>
#include "gpu/gpu_spatial.h"
#include "tbb/task_scheduler_init.h"

#define DIFF(t1, t2)    ((t2.tv_sec + ((float)t2.tv_usec)/1000000.0) - (t1.tv_sec + ((float)t1.tv_usec)/1000000.0))

extern void *thread_parser(void *param);
extern void *thread_builder(void *param);
extern void *thread_filter(void *param);
extern void *thread_aggregator(void *param);

void *(*thread_pointers[])(void *param) = {
	thread_parser,
	thread_builder,
	thread_filter,
	thread_aggregator
};

int main(int argc, char *argv[])
{
	pthread_t tid[4];
	struct timeval t1, t2;

	tbb::task_scheduler_init init(4);

	gettimeofday(&t1, NULL);
	int nr_gpus = get_gpu_device_count();
	gettimeofday(&t2, NULL);
	printf("time on the first cuda call: %lf s\n", DIFF(t1, t2));

//	nr_gpus = 1;

	init_device_streams(nr_gpus);

	//printf("Number of CUDA devices: %d\n", nr_gpus);
	if(nr_gpus < 1) {
		fprintf(stderr, "Error: at least one CUDA device is required to run this program\n");
		exit(1);
	}

	gettimeofday(&t1, NULL);

	for(int i = 0; i < 4; i++) {
		if(pthread_create(&tid[i], NULL, thread_pointers[i], (void *)&nr_gpus)) {
			fprintf(stderr, "failed to create thread %d\n", i);
			exit(1);
		}
	}

	for(int i = 0; i < 4; i++) {
		pthread_join(tid[i], NULL);
	}

	gettimeofday(&t2, NULL);
	printf("Time taken in pipeline: %lf s\n", DIFF(t1, t2));

	fini_device_streams();

	return 0;
}
