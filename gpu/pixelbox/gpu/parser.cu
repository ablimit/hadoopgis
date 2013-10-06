#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/stat.h>
#include <sys/time.h>
#include "gpu_spatial.h"

using namespace std;

#define THREAD1 128
#define BLOCK1 64
#define THREAD2 64
#define BLOCK2 128

extern cudaStream_t *stream;

__global__ void parseMBR(char *MBRRaw, int *MBR, int numOfCasts){
	for (int i=0; i<numOfCasts; i++) {
		int dstIndx = blockDim.x * blockIdx.x * numOfCasts + threadIdx.x + i * blockDim.x;
		int srcIndx = dstIndx * 5;
		MBR[dstIndx] = (MBRRaw[srcIndx+1] - '0') * 1000 + (MBRRaw[srcIndx + 2] - '0') * 100 + 
			(MBRRaw[srcIndx+3] - '0') * 10 + (MBRRaw[srcIndx + 4] - '0');
	}
}

__global__ void parseVertices(char *verticesRaw, int *offsetRaw, int* numOfVerticesInApoly, 
	int *X, int*Y, int *offsetInGPUMem, int nr_polys, int numOfPolyBlock){
	for (int i=0; i<numOfPolyBlock; i++) {
		int polyIndx = blockIdx.x + i*gridDim.x;
		if (polyIndx < nr_polys){
			int numOfVertices = numOfVerticesInApoly[polyIndx];
			int numOfVerticePerThread = (numOfVertices/blockDim.x + 1)*blockDim.x;
			int offsetInGPU = offsetInGPUMem[polyIndx];
			for (int j=0; j<numOfVerticePerThread; j++) {
				int verticeIndx = threadIdx.x + j*blockDim.x;
				if (verticeIndx < numOfVertices) {
					int srcIndx = offsetRaw[polyIndx] + verticeIndx*11;
					X[offsetInGPU + verticeIndx] = (verticesRaw[srcIndx+1] - '0') * 1000 + (verticesRaw[srcIndx + 2] - '0') * 100 + 
						(verticesRaw[srcIndx+3] - '0') * 10 + (verticesRaw[srcIndx + 4] - '0');
					Y[offsetInGPU + verticeIndx] = (verticesRaw[srcIndx+6] - '0') * 1000 + (verticesRaw[srcIndx + 7] - '0') * 100 + 
						(verticesRaw[srcIndx+8] - '0') * 10 + (verticesRaw[srcIndx + 9] - '0');
				}
			}
		}
	}
}

struct timespec diff(struct timespec start, struct timespec end){
	struct timespec temp;
	if ((end.tv_nsec-start.tv_nsec)<0) {
		temp.tv_sec = end.tv_sec-start.tv_sec-1;
		temp.tv_nsec = 1000000000+end.tv_nsec-start.tv_nsec;
	} else {
		temp.tv_sec = end.tv_sec-start.tv_sec;
		temp.tv_nsec = end.tv_nsec-start.tv_nsec;
	}
	return temp;
}

int getSizeBasedGPUConf1(int size){
	return (size/(THREAD1*BLOCK1) + 1)*THREAD1*BLOCK1;
}

int getSizeBasedGPUConf2(int size){
	return (size/THREAD2 + 1)*THREAD2;
}

int alloc_poly_array(poly_array_t *polys, const int nr_polys, const int nr_vertices)
{
	int size_mbrs = nr_polys * sizeof(mbr_t);
	int size_offsets = (nr_polys + 1) * sizeof(int);
	int size_x = nr_vertices * sizeof(int);
	int size_y = nr_vertices * sizeof(int);

	polys->mbrs = (mbr_t *)malloc(size_mbrs + size_offsets + size_x + size_y);
	if(!polys->mbrs) {
		fprintf(stderr, "failed to allocate memory for poly array\n");
		exit(1);
	}

	polys->nr_polys = nr_polys;
	polys->nr_vertices = nr_vertices;
	polys->offsets = (int *)((char *)(polys->mbrs) + size_mbrs);
	polys->x = (int *)((char *)(polys->offsets) + size_offsets);
	polys->y = (int *)((char *)(polys->x) + size_x);

	return 0;
}

poly_array_t *gpu_parse(int dno, char *file_name)
{
	char *MBRRaw, *verticesRaw, numOfVerticesBuf[4];
	char *dev_MBRRaw, *dev_verticesRaw;
	int *offset, *offsetRaw, *numOfVerticesInApoly;
	int *dev_offsetRaw, *dev_numOfVerticesInApoly;
	int *MBR, *X, *Y, *offsetInGPUMem;
	int *dev_MBR, *dev_X, *dev_Y, *dev_offsetInGPUMem;

	static const int parse_buf_size = 8192;
	char readbuf[parse_buf_size];
	fstream polyFile;
	int nr_polys, nr_vertices;

	polyFile.open(file_name, fstream::in | fstream::binary);

	polyFile.getline(readbuf, parse_buf_size);
	sscanf(readbuf, "%d, %d\n", &nr_polys, &nr_vertices);
//	cout <<"num of polygon: " <<nr_polys <<" , number of vertices:" <<nr_vertices<<endl;

	MBRRaw = (char *)malloc(getSizeBasedGPUConf1(20*nr_polys*sizeof(char)));
	MBR = (int *)malloc(getSizeBasedGPUConf1(4*nr_polys*sizeof(int)));

	cudaSetDevice(dno);

	cudaMalloc((void**)&dev_MBRRaw, getSizeBasedGPUConf1(20*nr_polys*sizeof(char))); 
	cudaMalloc((void**)&dev_MBR, getSizeBasedGPUConf1(4*nr_polys*sizeof(int))); 

	verticesRaw = (char *)malloc(nr_vertices * 11);
	offset = (int *)malloc(nr_polys*sizeof(int));
	offsetRaw = (int *)malloc(nr_polys*sizeof(int));
	numOfVerticesInApoly = (int *)malloc(nr_polys*sizeof(int));
	offsetInGPUMem = (int *)malloc(nr_polys*sizeof(int));

	cudaMalloc((void**)&dev_verticesRaw, nr_vertices * 11); 
	cudaMalloc((void**)&dev_offsetRaw, nr_polys*sizeof(int)); 
	cudaMalloc((void**)&dev_numOfVerticesInApoly, nr_polys*sizeof(int)); 
	cudaMalloc((void**)&dev_offsetInGPUMem, nr_polys*sizeof(int)); 

	int rawBufferIndx = 0;
	int vertexIndx = 0;
	int numVertices = 0;
	int numVerticesInGPUMem = 0;
	/* process first vertex line*/
	polyFile.getline(readbuf, parse_buf_size);
	memcpy(numOfVerticesBuf, readbuf, 4);
	memcpy(MBRRaw, readbuf + 5, 20);
	offset[0] = vertexIndx;
	offsetRaw[0] = rawBufferIndx;
	numVertices = atoi(numOfVerticesBuf);
	offsetInGPUMem[0] = numVerticesInGPUMem;
	numVerticesInGPUMem = (numVertices/THREAD2 + 1) * THREAD2;
	numOfVerticesInApoly[0] = numVertices;
	memcpy(verticesRaw, readbuf + 26, numVertices * 11);
	vertexIndx += numVertices;
	rawBufferIndx += strlen(readbuf)-26;
	//cout<<atoi(numOfVerticesBuf)<<endl;

	/* each iteration process a line */
	for (int i=1; i<nr_polys; i++) {
		polyFile.getline(readbuf, parse_buf_size);
		memcpy(numOfVerticesBuf, readbuf, 4);
		memcpy(MBRRaw + 20*i, readbuf + 5, 20);
		offset[i] = vertexIndx;
		offsetRaw[i] = rawBufferIndx;
		numVertices = atoi(numOfVerticesBuf);
		offsetInGPUMem[i] = numVerticesInGPUMem;
		numVerticesInGPUMem += (numVertices/THREAD2 + 1) * THREAD2;
		numOfVerticesInApoly[i] = numVertices;
		memcpy(verticesRaw + rawBufferIndx, readbuf + 26, numVertices * 11);
		vertexIndx += numVertices;
		rawBufferIndx += strlen(readbuf)-26;
		//break;
		//cout<<atoi(numOfVerticesBuf)<<endl;
	}

	//cout<<getSizeBasedGPUConf2(nr_vertices*sizeof(int))<<" "<<(nr_vertices*sizeof(int)/THREAD2 + 1 )*THREAD2
	//	<<" "<<numVerticesInGPUMem<<" "<<nr_polys/BLOCK2 + 1<<" "<<rawBufferIndx<<endl;
	//cout<<verticesRaw + offsetRaw[nr_polys-1]<<endl;
	//cout<<verticesRaw<<endl;
	X = (int *)malloc(nr_vertices * sizeof(int));
	Y = (int *)malloc(nr_vertices * sizeof(int));
	cudaMalloc((void**)&dev_X, nr_vertices * sizeof(int)); 
	cudaMalloc((void**)&dev_Y, nr_vertices * sizeof(int)); 

	//cout<<MBRRaw<<endl<<endl;
	/* for debugging
	cout<<offset[0]<<" "<<offset[1]<<endl;
	cout<<offsetRaw[0]<<" "<<offsetRaw[1]<<endl;
	cout<<MBRRaw<<endl<<endl;
	cout<<verticesRaw<<endl;
	*/
	//cout<<"number per THREAD1 "<<getSizeBasedGPUConf1(4*nr_polys*sizeof(int))/(BLOCK1*THREAD1)<<endl;

	cudaMemcpy(dev_MBRRaw, MBRRaw, getSizeBasedGPUConf1(20*nr_polys*sizeof(char)),  cudaMemcpyHostToDevice);

	cudaEvent_t start, stop;
	float elapsedTime;
	cudaEventCreate( &start );
    cudaEventCreate( &stop );
	cudaEventRecord( start, 0 );
	parseMBR<<<BLOCK1, THREAD1, 0, stream[dno]>>>( dev_MBRRaw, dev_MBR, getSizeBasedGPUConf1(nr_polys*4)/(BLOCK1*THREAD1));
	cudaThreadSynchronize();

//	cudaEventRecord( stop, 0 );
//	cudaEventSynchronize( stop );
//	cudaEventElapsedTime( &elapsedTime, start, stop );
//	cout<<"Time:" <<elapsedTime<<"ms"<<endl;				
//	cudaEventDestroy( start );
//  cudaEventDestroy( stop );

	cudaMemcpy(MBR, dev_MBR, getSizeBasedGPUConf1(4*nr_polys*sizeof(int)),  cudaMemcpyDeviceToHost); 

	cudaMemcpy(dev_verticesRaw, verticesRaw, nr_vertices * 11,  cudaMemcpyHostToDevice); 
	cudaMemcpy(dev_offsetRaw, offsetRaw, nr_polys*sizeof(int),  cudaMemcpyHostToDevice); 
	cudaMemcpy(dev_numOfVerticesInApoly, numOfVerticesInApoly, nr_polys*sizeof(int),  cudaMemcpyHostToDevice); 
	cudaMemcpy(dev_offsetInGPUMem, offset, nr_polys*sizeof(int),  cudaMemcpyHostToDevice);
//	cudaEventCreate( &start );
//    cudaEventCreate( &stop );
//	cudaEventRecord( start, 0 );
	parseVertices<<<BLOCK2, THREAD2, 0, stream[dno]>>>(dev_verticesRaw, dev_offsetRaw, dev_numOfVerticesInApoly, 
		dev_X, dev_Y, dev_offsetInGPUMem, nr_polys, nr_polys/BLOCK2 + 1);
//	cudaThreadSynchronize();
//	cudaEventRecord( stop, 0 );
//	cudaEventSynchronize( stop );
//	cudaEventElapsedTime( &elapsedTime, start, stop );
//	cout<<"Time:" <<elapsedTime<<"ms"<<endl;

	cudaMemcpy(X, dev_X, nr_vertices * sizeof(int),  cudaMemcpyDeviceToHost); 
	cudaMemcpy(Y, dev_Y, nr_vertices * sizeof(int),  cudaMemcpyDeviceToHost); 

	poly_array_t *polys = (poly_array_t *)malloc(sizeof(poly_array_t));
	alloc_poly_array(polys, nr_polys, nr_vertices);

	memcpy(polys->mbrs, MBR, sizeof(mbr_t) * nr_polys);
	memcpy(polys->offsets, offset, sizeof(int) * nr_polys);
	polys->offsets[nr_polys] = nr_vertices;
	memcpy(polys->x, X, sizeof(int) * nr_vertices);
	memcpy(polys->y, Y, sizeof(int) * nr_vertices);

	//int ret_X[nr_vertices], ret_Y[nr_vertices];
	//for (int i=0; i<nr_polys; i++) {
	//	memcpy(&ret_X[offset[i]], &X[offsetInGPUMem[i]], numOfVerticesInApoly[i]*sizeof(int));
	//	memcpy(&ret_Y[offset[i]], &Y[offsetInGPUMem[i]], numOfVerticesInApoly[i]*sizeof(int));
	//}

	//cout<<" "<<X[nr_vertices - 1]<<" "<<Y[nr_vertices - 1];
//	for (int i=0; i<nr_polys; i++) {
//		cout<<numOfVerticesInApoly[i]<<" "<<MBR[4*i]<<" "<<MBR[4*i+1]<<" "<<MBR[4*i+2]<<" "<<MBR[4*i+3];
//		for (int j=0; j<numOfVerticesInApoly[i]; j++){
//			cout<<" "<<X[offset[i]+j]<<" "<<Y[offset[i]+j];
//		}
//		cout<<endl;
//	}

	free(MBRRaw);
	free(verticesRaw);
	free(offset);
	free(offsetRaw);
	free(numOfVerticesInApoly);
	free(offsetInGPUMem);
	free(MBR);
	free(X);
	free(Y);
	cudaFree(dev_MBRRaw);
	cudaFree(dev_MBR);
	cudaFree(dev_verticesRaw);
	cudaFree(dev_offsetRaw);
	cudaFree(dev_numOfVerticesInApoly);
	cudaFree(dev_offsetInGPUMem);
	cudaFree(dev_X);
	cudaFree(dev_Y);

	return polys;
}
