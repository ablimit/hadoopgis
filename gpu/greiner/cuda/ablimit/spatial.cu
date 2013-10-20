#include <cstdlib>
#include <cstring>
#include <cstdio>

//	Calculates the euclidean distance between (x1, y1) and (x2, y2)
__device__ float Dist(float x1, float y1, float x2, float y2)
{
	return sqrtf((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)); 
}

//	Calculates the intersection between the two line segments p and q.
__device__ int SegmentIntersect(
	point_type * p1x,
	point_type * p1y,
	point_type * p2x,
	point_type * p2y,
	point_type * q1x,
	point_type * q1y,
	point_type * q2x,
	point_type * q2y,
	point_type *xInt,
	point_type *yInt, 
	float *alphaP, float *alphaQ)
{
    float det;

    *yInt = *xInt = *alphaP = *alphaQ = -1.0f;

    // Check if lines are parallel
    det = (p2x - p1x) * (q2y - q1y) - (p2y - p1y) * (q2x - q1x);

    if( det == 0 ) 
	return 0;

    // Lines are not parallel

    float tp, tq;
    // Check if the segments actually intersect	
    tp = ((q1x - p1x) * (q2y - q1y) - (q1y - p1y) * (q2x - q1x)) / det; 
    tq = ((p2y - p1y) * (q1x - p1x) - (p2x - p1x) * (q1y - p1y)) / det;

    if(tp<0 || tp>1 || tq<0 || tq>1) return 0;
    // Line segments intersect

    // Calculate the actual intersection
    *xInt = p1x + tp * (p2x - p1x);
    *yInt = p1y + tp * (p2y - p1y);

    *alphaP = Dist(p1x, p1y, *xInt, *yInt) / Dist(p1x, p1y, p2x, p2y); 
    *alphaQ = Dist(q1x, q1y, *xInt, *yInt) / Dist(q1x, q1y, q2x, q2y); 
    return 1;
}


// calculate the line segment intersection between set of polygons

int gpuIntersect(point_type *subj, int * polygon_idx , VERTEX *clip, int clipSize,
	VERTEX **intPoly, int *intPolySize)
{
    int		result = 1;
    vertex	*devSubj, *devClip, *devIntSubj, *devIntClip,
		*intClip, *intSubj;
    int		*devSize, size[2];
    cudaError_t	devResult;


    fprintf(stderr,"AllocateDevMem-----");
    result = AllocateDevMem(devSize, devClip, devSubj, devIntClip, devIntSubj, clipSize, subjSize);

    result ? fprintf(stderr,"success.\n"): fprintf(stderr,"fail\n");
    fprintf(stderr,"AllocateIntBuffers-----");

    if( result )
	result = AllocateIntBuffers(&intClip, &intSubj, clipSize, subjSize);

    result ? fprintf(stderr,"success.\n"): fprintf(stderr,"fail\n");
    fprintf(stderr,"CopyData subj-----");

    if( result ) 
	result = CopyData(devSubj, subj, subjSize * sizeof(vertex), cudaMemcpyHostToDevice);

    result ? fprintf(stderr,"success.\n"): fprintf(stderr,"fail\n");
    fprintf(stderr,"CopyData clip-----");

    if( result )
	result = CopyData(devClip, clip, clipSize * sizeof(vertex), cudaMemcpyHostToDevice);

    result ? fprintf(stderr,"success.\n"): fprintf(stderr,"fail\n");

    fprintf(stderr,"CalcIntersection-----");

    // Calulate the intersection points.
    if( result ) {

	CalcIntersections<<<1, 128>>>(devClip, clipSize, devSubj, subjSize,
		devIntClip, devIntSubj, devSize);

	devResult = cudaMemcpy(size, devSize, 2 * sizeof(int), cudaMemcpyDeviceToHost);
	if( devResult != cudaSuccess ) {
	    result = 0;
	}
	result ? fprintf(stderr,"success.\n"): fprintf(stderr,"fail\n");

    } else

	if( result ) 
	    result = CopyData(intSubj, devIntSubj, clipSize * subjSize * sizeof(vertex), cudaMemcpyDeviceToHost);

    if( result ) 
	result = CopyData(intClip, devIntClip, clipSize * subjSize * sizeof(vertex), cudaMemcpyDeviceToHost);

    for (int i =0 ; i< clipSize * subjSize; i++)
	fprintf(stderr,"(%d,%d)\n",(int)intSubj[i].x,(int)intSubj[i].y);

    fprintf(stderr,"\n");

    for (int i =0 ; i< clipSize * subjSize; i++)
	fprintf(stderr,"(%d,%d)\n",(int)intClip[i].x,(int)intClip[i].y);

    exit(1);
    fprintf(stderr,"BuildIntPoly-----");

    if( result ) {

	result = BuildIntPoly(*intPoly, *intPolySize, intSubj, size[SUBJ], intClip, size[CLIP]);
    }

    result ? fprintf(stderr,"success.\n"): fprintf(stderr,"fail\n");
    fprintf(stderr,"FreeDevMem-----");

    result = FreeDevMem(&devClip, &devSubj, &devIntClip, &devIntSubj);

    result ? fprintf(stderr,"success.\n"): fprintf(stderr,"fail\n");

    return result;
}



