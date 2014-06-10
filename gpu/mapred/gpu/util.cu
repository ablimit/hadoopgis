#include <iostream>
#include "gpu_refine.h"

using namespace std;

cudaStream_t *stream = NULL;

int get_gpu_device_count(void)
{
	int count;
	cudaGetDeviceCount(&count);
	return count;
}

void init_device_streams(int nr_devices)
{
	stream = new cudaStream_t[nr_devices];
	if(!stream) {
		cerr <<"failed to allocate cuda streams" <<endl;
		exit(1);
	}

	for(int i = 0; i < nr_devices; i++) {
		cudaSetDevice(i);
		cudaStreamCreate(&stream[i]);
	}
}

void fini_device_streams(int nr_devices)
{
  for(int i = 0; i < nr_devices; i++) {
    cudaStreamDestroy(stream[i]);
  }
  delete [] stream;
}

