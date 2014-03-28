#include <iostream>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/stat.h>
#include <sys/time.h>
//#include "gpu_spatial.h"
#include "parser.cuh"

using namespace std;

#define THREAD1 128
#define BLOCK1 128
#define THREAD2 64
#define BLOCK2 128

extern cudaStream_t *stream;

__global__ void parseMBR(char *MBRRaw, int *MBR, int nr_polys){
  int baseIdx = blockDim.x * blockIdx.x + threadIdx.x ;
  if (baseIdx < nr_polys) { 
    baseIdx = baseIdx * 4 ;
    for (int i=0; i< 4; i++) {
      int dstIndx = baseIdx + i ;
      int srcIndx = dstIndx * 5;
      MBR[dstIndx] = (MBRRaw[srcIndx] - '0') * 1000 + (MBRRaw[srcIndx + 1] - '0') * 100 + 
        (MBRRaw[srcIndx+2] - '0') * 10 + (MBRRaw[srcIndx + 3] - '0');
    }
  }
}

__global__ void parseVertices(char *verticesRaw, int *offsetRaw, int* numOfVerticesInApoly, 
    int *X, int*Y, int *offset, int nr_polys){
  
  int polyIdx    = blockDim.x * blockIdx.x + threadIdx.x ;  // 1 polygon / thread 

  if (polyIdx < nr_polys){  
    int numOfVertices = numOfVerticesInApoly[polyIdx];
    int raw_offset = offsetRaw[polyIdx];  // raw offset in GPU memory 
    int ver_offset = offset[polyIdx];    // vertex offset in GPU memory 

    for (int j=0; j<numOfVertices; j++) {
      int srcIndx = raw_offset + j*10;
      X[ver_offset + j ] = (verticesRaw[srcIndx] - '0') * 1000 + (verticesRaw[srcIndx + 1] - '0') * 100 + 
        (verticesRaw[srcIndx+2] - '0') * 10 + (verticesRaw[srcIndx + 3] - '0');
      Y[ver_offset + j ] = (verticesRaw[srcIndx + 5] - '0') * 1000 + (verticesRaw[srcIndx + 6 ] - '0') * 100 + 
        (verticesRaw[srcIndx + 7 ] - '0') * 10 + (verticesRaw[srcIndx + 8 ] - '0');
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

poly_array_t *gpu_parse(int dno, const int nv, std::vector<string> * poly_vec) 
{
  char *MBRRaw, *verticesRaw, numOfVerticesBuf[4];
  char *dev_MBRRaw, *dev_verticesRaw;
  int *offset, *offsetRaw, *numOfVerticesInApoly;
  int *dev_offset, *dev_offsetRaw, *dev_numOfVerticesInApoly;
  int *MBR, *X, *Y ;
  int *dev_MBR, *dev_X, *dev_Y;

  static const int BUF_SIZE = 8192;
  char buf[BUF_SIZE];
  int nr_polys = poly_vec->size();  // number of polygons
  int nr_vertices = nv; // number of vertices 

  //std::cerr << "inGPU poly parsing ..." << std::endl; 
  //	cout <<"num of polygon: " <<nr_polys <<" , number of vertices:" <<nr_vertices<<endl;

  MBRRaw = (char *)malloc(20*nr_polys*sizeof(char));
  MBR = (int *)malloc(4*nr_polys*sizeof(int));

  cudaSetDevice(dno);

  cudaMalloc((void**)&dev_MBRRaw, 20*nr_polys*sizeof(char)); 
  cudaMalloc((void**)&dev_MBR, 4*nr_polys*sizeof(int)); 

  verticesRaw = (char *)malloc(nr_vertices * 10);
  offset = (int *)malloc(nr_polys*sizeof(int));
  offsetRaw = (int *)malloc(nr_polys*sizeof(int));
  numOfVerticesInApoly = (int *)malloc(nr_polys*sizeof(int));

  cudaMalloc((void**)&dev_verticesRaw, nr_vertices * 10); 
  cudaMalloc((void**)&dev_offsetRaw, nr_polys*sizeof(int)); 
  cudaMalloc((void**)&dev_numOfVerticesInApoly, nr_polys*sizeof(int)); 
  cudaMalloc((void**)&dev_offset, nr_polys*sizeof(int)); 

  int rawBufferIndx = 0;
  int vertexIndx = 0;
  int numVertices = 0;

  /* each iteration process a line */
  std::size_t len =0 ;
  for (int i= 0 ; i<nr_polys; i++) {
    len = (*poly_vec)[i].copy(buf, BUF_SIZE); 
    buf[len] = '\0';
    memcpy(numOfVerticesBuf, buf, 4); // 
    memcpy(MBRRaw + 20*i, buf + 5, 20);
    offset[i] = vertexIndx;
    offsetRaw[i] = rawBufferIndx;
    numVertices = atoi(numOfVerticesBuf);
    numOfVerticesInApoly[i] = numVertices;
    memcpy(verticesRaw + rawBufferIndx, buf + 25, numVertices * 10);
    vertexIndx += numVertices;
    rawBufferIndx += len - 25;
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

  cudaMemcpy(dev_MBRRaw, MBRRaw, 20*nr_polys*sizeof(char),  cudaMemcpyHostToDevice);

  /* timing utilities 
  cudaEvent_t start, stop;
  float elapsedTime;
  cudaEventCreate( &start );
  cudaEventCreate( &stop );
  cudaEventRecord( start, 0 );
  */
  parseMBR<<<BLOCK1, THREAD1, 0, stream[dno]>>>(dev_MBRRaw, dev_MBR, nr_polys);
  cudaThreadSynchronize();
  
  /*
  cudaEventRecord( stop, 0 );
  cudaEventSynchronize( stop );
  cudaEventElapsedTime( &elapsedTime, start, stop );
  cerr <<"Time to parse MBR: " <<elapsedTime<<"ms"<<endl;				
  cudaEventDestroy( start );
  cudaEventDestroy( stop );
  */
  cudaMemcpy(MBR, dev_MBR, 4*nr_polys*sizeof(int),  cudaMemcpyDeviceToHost); 

  /* debug purpose  
   *for (int i=0; i<nr_polys; i++) {
   *  cout<<MBR[4*i]<<" "<<MBR[4*i+1]<<" "<<MBR[4*i+2]<<" "<<MBR[4*i+3] << endl;
   *}
   */

  cudaMemcpy(dev_verticesRaw, verticesRaw, nr_vertices * 10,  cudaMemcpyHostToDevice); 
  cudaMemcpy(dev_offsetRaw, offsetRaw, nr_polys*sizeof(int),  cudaMemcpyHostToDevice); 
  cudaMemcpy(dev_numOfVerticesInApoly, numOfVerticesInApoly, nr_polys*sizeof(int),  cudaMemcpyHostToDevice); 
  cudaMemcpy(dev_offset, offset, nr_polys*sizeof(int),  cudaMemcpyHostToDevice);
  /* cudaEventCreate( &start );
  cudaEventCreate( &stop );
  cudaEventRecord( start, 0 );
  */
  parseVertices<<<BLOCK1, THREAD1, 0, stream[dno]>>>(dev_verticesRaw, dev_offsetRaw, dev_numOfVerticesInApoly, 
      dev_X, dev_Y, dev_offset, nr_polys);
  cudaThreadSynchronize();
  
  /* 
  cudaEventRecord( stop, 0 );
  cudaEventSynchronize( stop );
  cudaEventElapsedTime( &elapsedTime, start, stop );
  cerr <<"Time to parse POLY: " <<elapsedTime<<"ms"<<endl;
  */

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
  /* for (int i=0; i<nr_polys; i++) {
    cout<<numOfVerticesInApoly[i] ; // <<MBR[4*i]<<" "<<MBR[4*i+1]<<" "<<MBR[4*i+2]<<" "<<MBR[4*i+3];
    for (int j=0; j<numOfVerticesInApoly[i]; j++){
      cout<<", "<<X[offset[i]+j]<<" "<<Y[offset[i]+j];
    }
    cout<<endl;
  }*/

  free(MBRRaw);
  free(verticesRaw);
  free(offset);
  free(offsetRaw);
  free(numOfVerticesInApoly);
  free(MBR);
  free(X);
  free(Y);
  cudaFree(dev_MBRRaw);
  cudaFree(dev_MBR);
  cudaFree(dev_verticesRaw);
  cudaFree(dev_offsetRaw);
  cudaFree(dev_numOfVerticesInApoly);
  cudaFree(dev_offset);
  cudaFree(dev_X);
  cudaFree(dev_Y);

  return polys;
}
