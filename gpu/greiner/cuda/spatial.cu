//
//
//	gpu-poly	
//
//	
//		Polygon functions for the GPU
//	
#include <cstdlib>
#include <cstring>
#include <cstdio>

#include "spatial.cuh"


#define	CLIP	0
#define	SUBJ	1

//
//	Determines of the specified point is in the specified polygon
//
//
__device__ bool PointInPoly(vertex point, vertex *poly, int polySize)
{
	bool	inPoly = false;
	int		i, j;
	
	for(i = 0, j = polySize - 1; i < polySize; j = i++) {
		if( ((poly[i].y > point.y) != (poly[j].y > point.y)) &&
			 (point.x < (poly[j].x - poly[i].x) * ( point.y - poly[i].y) /
			 						 (poly[j].y - poly[i].y) + poly[i].x) ) {
			 						 
			 inPoly = !inPoly;
		}
	}
	return inPoly;
}





//
// 	Trace the subject polygon and determine the vertices interior
//	to the clip polygon using the even/odd rule. In other words, 
//	determine if the first vertex of subj is inside the clip poly,
//	then toggle internal every time an intersection vertex is 
//	encountered. (Intersection vertices have a non-zero alpha value)
//
__device__ void MarkEntry(vertex *clip, int clipSize, vertex *subj, int subjSize)
{
	bool	in = PointInPoly(subj[0], clip, clipSize);
	int		i = subj[0].next;
	
	subj[0].internal = in;
	
	while( i < subjSize ) {

		if( subj[i].alpha != 0.0f ) {
			in = !in; 
		}
		
		// Need to mark exit points also
		if( subj[i].alpha != 0.0f && in == false ) 
			subj[i].internal = !in;
		else
			subj[i].internal = in;
		
		i = subj[i].next;
	}
}





// 
// 	Combine the points and arrange the intersection points by alpha value.
//	Note!!! - This function needs to be called by a single thread. There's 
// 			  probably a better way to do this...//
//
__device__ void CombinePoints(vertex *newArray, vertex *srcArray, 
							  int srcSize, int &newSize)
{
	int		index = 0, srcIdx = 0, next, prev, base = 0;
	newSize = 0;
	
	// Hold on to index of last non-intersect point... Place new point at
	// the end of the array, Adjust next index of intersection points.
	while( srcIdx < srcSize ) {
		if( srcArray[srcIdx].x != -1 ) {

			// Copy to new array in order.
			newArray[index] = srcArray[srcIdx++];
			newArray[index].next = index + 1;
			newSize++;
			
			if( newArray[index].alpha != 0 ) {
				// Intersection point, set index to be in alpha order.
				// Start at base and insert new point in appropriate spot.
				next = newArray[base].next;
				prev = base;

				while( next < index ) {
					
					if( newArray[index].alpha < newArray[next].alpha ) {
					
						newArray[prev].next = index;
						newArray[index].next = next;
						newArray[index - 1].next = index + 1;
						break;
					}
					prev = next;
					next = newArray[next].next;
				}
			} else {
				// Not an intersection point, use as next base
				base = index;
			}
			index++;
		} else 
			srcIdx++;
	}
}





//
//	Adjusts the link to place intersection points in the proper order. Also
//	skip over empty vertices. Changed to this method so that the linkTag will
//	refer to the proper vertex of the other polygon when extracting the intersecting
//	or union polygon.
//
//		NOTE! This function needs to be called with a single thread
//
void __device__ LinkPoints(vertex *verts, int size, int stride)
{
	int 	index = 0, next, prev, base = 0, last = 0;

	// Hold on to index of last non-intersect point... Insert index of 
	// intersection point in the proper place...
	while(index < size ) {
		if( verts[index].x != -1 ) {

			verts[last].next = index;
			
			if( verts[index].alpha != 0 ) {
				// Intersection point, adjust next pointer
				next = verts[base].next;
				prev = base;
				
				while( next < index ) {
				
					if( verts[index].alpha < verts[next].alpha ) {
					
						verts[prev].next = index;
						verts[index].next = next;
						break;
					}
					prev = next;
					next = verts[next].next;
				}
				if( next >= index )
					last = index;

			} else {
				// Not an intersection point, use as base.
				base = index;
				last = index;
			}
		} 
		index++;
	}
	verts[size - 1].next = size;
}


				

//
//	Calculates the euclidean distance between (x1, y1) and
//	(x2, y2)
//
//
__device__ float Dist(float x1, float y1, float x2, float y2)
{
	return sqrtf((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)); 
}





//
//	Calculate the intersection between the two line segments p and q.
// 	The alpha value is a measure between 0 and 1 that indicated the 
//	distance from the respective point. Used to order the intersection
//	points.
//	Note! - This function is called by only one thread.
//	
//
__device__ void Intersect(vertex *p1, vertex *p2, vertex *q1, vertex *q2,
				float *xInt, float *yInt, float *alphaP, float *alphaQ)
{
	float det;
	
	*yInt = *xInt = *alphaP = *alphaQ = -1.0f;
	
	// Check if lines are parallel
	det = (p2->x - p1->x) * (q2->y - q1->y) -
		  (p2->y - p1->y) * (q2->x - q1->x);
		  
	if( det != 0 ) {
		// Lines are not parallel
		
		float tp, tq;
		// Check if the segments actually intersect	
		tp = ((q1->x - p1->x) * (q2->y - q1->y) - (q1->y - p1->y) * (q2->x - q1->x)) / det; 
		tq = ((p2->y - p1->y) * (q1->x - p1->x) - (p2->x - p1->x) * (q1->y - p1->y)) / det;

		if( tp >= 0 && tp <= 1 && tq >= 0 && tq <= 1 ) {
			// Line segments intersect
			
			// Calculate the actual intersection
			*xInt = p1->x + tp * (p2->x - p1->x);
			*yInt = p1->y + tp * (p2->y - p1->y);
			
			*alphaP = Dist(p1->x, p1->y, *xInt, *yInt) / Dist(p1->x, p1->y, p2->x, p2->y); 
 			*alphaQ = Dist(q1->x, q1->y, *xInt, *yInt) / Dist(q1->x, q1->y, q2->x, q2->y); 
		}		 
	}
}





//
//	Calculates the intersection, if it exists, of each pair of edges from
//	clip and subj. Returns two polygons consisting of the original vertices
//	and the intersection points. 
//
//
__global__ void CalcIntersections(vertex *clip, int clipSize, vertex *subj, int subjSize,
								  vertex *newClip, vertex *newSubj, int *newPolySizes)
{
	int		tid = threadIdx.x + blockIdx.x * blockDim.x, 
			subjStart, clipStart, index;
	float	xInt, yInt, alphaSubj, alphaClip;

	// Calculate the intersections between clip and subj polygons
	while( tid < (subjSize * clipSize)  ) {
	
		if( (tid % clipSize) == 0 ) {
			newSubj[tid] = subj[tid / clipSize];
		} else {
			subjStart = tid / clipSize;
			clipStart = ((tid % clipSize) - 1);
			index = ( clipStart * subjSize) + (tid / clipSize) + 1;

			Intersect(&subj[subjStart], &subj[subjStart + 1], 
					  &clip[clipStart], &clip[clipStart + 1],
					  &xInt, &yInt, &alphaSubj, &alphaClip);

			newSubj[tid].x = xInt;
			newSubj[tid].y = yInt;
			newSubj[tid].next = -1;
			newSubj[tid].alpha = alphaSubj;			 
			
			newClip[index].x = xInt;
			newClip[index].y = yInt;
			newClip[index].next = -1;
			newClip[index].alpha = alphaClip;
			
			// Link the the two polygons at the intersection
			if( xInt != -1 ) {
				newSubj[tid].linkTag = index;
				newClip[index].linkTag = tid;
			} 
		}				
		
		if( (tid % subjSize) == 0 ) {
			newClip[tid] = clip[tid / subjSize];
		}
		
		tid += blockDim.x * gridDim.x;
	} 
	
	tid = threadIdx.x + blockIdx.x * blockDim.x;
	
	// Cleanup and order the intersections
	if( tid == 0 ) {
		//CombinePoints(newClip, newClip, ((clipSize - 1) * subjSize) + 1, newPolySizes[0]);
		newPolySizes[CLIP] = ((clipSize - 1) * subjSize) + 1;
		LinkPoints(newClip, newPolySizes[CLIP], subjSize);
	}
	if( tid == 1 ) {
		//CombinePoints(newSubj, newSubj, ((subjSize - 1) * clipSize) + 1, newPolySizes[1]);
		newPolySizes[SUBJ] = ((subjSize - 1) * clipSize) + 1;
		LinkPoints(newSubj, newPolySizes[SUBJ], clipSize);
	}	
	
	if( tid == 0 ) {
		MarkEntry(newClip, newPolySizes[CLIP], newSubj, newPolySizes[SUBJ]);		
	}
	
	if( tid == 1 ) {
		MarkEntry(newSubj, newPolySizes[SUBJ], newClip, newPolySizes[CLIP]);
	}
}





//
// 	Allocates device memory for the 4 polygons. Clip, Subj
// 	clip w/ intersections, subj w/ intersections.
//
bool AllocateDevMem(int *&size, vertex *&clip, vertex *&subj, vertex *&intClip, vertex *&intSubj,
					int clipSize, int subjSize)
{
	bool 			result = true;
	cudaError_t		devResult;	
	
	
	devResult = cudaMalloc((void**)&clip, clipSize * sizeof(vertex));
	if( devResult != cudaSuccess ) {
		result = false;
	}

	if( result ) {
		devResult = cudaMalloc((void**)&subj, subjSize * sizeof(vertex));
		if( devResult != cudaSuccess ) {
			result = false;
		}
	}

	if( result ) {
		devResult = cudaMalloc((void**)&intSubj, subjSize * clipSize * sizeof(vertex));
		if( devResult != cudaSuccess ) {
			result = false;
		}
	}

	if( result ) {
		devResult = cudaMalloc((void**)&intClip, subjSize * clipSize * sizeof(vertex));
		if( devResult != cudaSuccess ) {
			result = false;
		}
	}
	
	if( result ) {
		devResult = cudaMalloc((void**)&size, 2 * sizeof(int));
		if( devResult != cudaSuccess ) {
			result = false;
		}
	}		
	return result;
}





bool FreeDevMem(vertex **clip, vertex **subj, vertex **intClip, vertex **intSubj)
{
	cudaFree(*clip);
	*clip = NULL;
	cudaFree(*subj);
	*subj = NULL;
	cudaFree(*intClip);
	*intClip = NULL;
	cudaFree(*intSubj);
	*intSubj = NULL;

	return true;
}






bool AllocateIntBuffers(vertex **clip, vertex **subj, int clipSize, int subjSize)
{
	bool	result = true;

	*clip = (vertex*)malloc(clipSize * subjSize * sizeof(vertex));
	if( *clip == NULL ) {
		result = false;
	}
	
	if( result ) {
		*subj = (vertex*)malloc(clipSize * subjSize * sizeof(vertex));
		if( *subj == NULL ) {
			result = false;
		}
	}
	return result;
}





bool CopyData(vertex *dest, vertex *src, int size, enum cudaMemcpyKind dir)
{
	bool			result = true;
	cudaError_t		devResult;
	
	devResult = cudaMemcpy(dest, src, size, dir);
						   
	if( devResult != cudaSuccess )
		result = false;
	return  result;
}





//
//	Adds a vertex to the specified polygon by reallocating the 
//	buffer. May want to change this to work with chunks of memory
//	rather than a single vertex at a time.
//
bool AddVertex(vertex *&poly, int& size, vertex *vert) 
{
	bool 	result = true;
	vertex 	*temp = (vertex*)realloc(poly, (size + 1) * sizeof(vertex));
	
	if( temp != NULL ) {
		poly = temp;
		poly[size++] = *vert;
	} else {
		result = false;
	}
	return result;
}




		
//	
//	Builds the intersection polygon by tracing the subject
//	polygon from the first intersection point. When the next
//	vertex is not internal we jump to the corresponding vertex
//	in the clip polygon and continue form there. This pattern is
//	continued (back and forth between clip & subj for internal
//	vertices) until we reach the starting vertex.
//
//
bool BuildIntPoly(vertex *&intPoly, int& intPolySize, 
			 vertex *subj, int subjSize,
			 vertex *clip, int clipSize)
{
	bool	result = true, inSubj = true;
	vertex	*curVert = subj, *firstVert;
	
	// Find first intersection point in the subj poly
	while( curVert->internal == false ) {
		curVert = &subj[curVert->next];
	}
	firstVert = curVert;
	if( AddVertex(intPoly, intPolySize, curVert) ) {
	
		do {
			
			if( inSubj ) {
				if( subj[curVert->next].internal ) {

					curVert = &subj[curVert->next];
				} else {
					
					curVert = &clip[clip[curVert->linkTag].next];
					inSubj = false;
				}
			} else {
				if( clip[curVert->next].internal ) {
					curVert = &clip[curVert->next];
				} else {
					curVert = &subj[subj[curVert->linkTag].next];
					inSubj = true;
				}			
			}

			if( !AddVertex(intPoly, intPolySize, curVert) ) {
				result = false;
				break;
			}
		} while( !(curVert->x == firstVert->x &&
				   curVert->y == firstVert->y) );
		
	} else {
		result = false;
	}
	return result;
}





//	
//	Builds the union polygon similar to the way the
//	intersection polygon is built. Though instead of
//	looking for internal vertices, we want external ones.
//
//
bool BuildUnionPoly(vertex *&unionPoly, int& unionPolySize, 
			 vertex *subj, int subjSize,
			 vertex *clip, int clipSize)
{
	bool	result = true, inSubj = true;
	vertex	*curVert = subj, *firstVert;
	
	// Find first intersection point in the subj poly
	while( curVert->internal == false ) {
		curVert = &subj[curVert->next];
	}
	firstVert = curVert;
	if( AddVertex(unionPoly, unionPolySize, curVert) ) {
	
		do {
			
			if( inSubj ) {
				if( !subj[curVert->next].internal ) {

					curVert = &subj[curVert->next];
				} else {
					
					curVert = &clip[clip[curVert->linkTag].next];
					inSubj = false;
				}
			} else {
				if( !clip[curVert->next].internal ) {
					curVert = &clip[curVert->next];
				} else {
					curVert = &subj[subj[curVert->linkTag].next];
					inSubj = true;
				}			
			}

			if( !AddVertex(unionPoly, unionPolySize, curVert) ) {
				result = false;
				break;
			}
		} while( !(curVert->x == firstVert->x &&
				   curVert->y == firstVert->y) );
		
	} else {
		result = false;
	}
	return result;
}






// 
//	Calculate the intersection polygon
//
//
int gpuIntersect(VERTEX *subj, int subjSize, VERTEX *clip, int clipSize,
				 VERTEX **intPoly, int *intPolySize)
{
	int		result = 1;
	vertex	*devSubj, *devClip, *devIntSubj, *devIntClip,
			*intClip, *intSubj;
	int		*devSize, size[2];
	cudaError_t	devResult;

		
	printf("AllocateDevMem-----");
	result = AllocateDevMem(devSize, devClip, devSubj, devIntClip, devIntSubj, clipSize, subjSize);
	
	result ? printf("success.\n"): printf("fail\n");
	printf("AllocateIntBuffers-----");
	
	if( result )
		result = AllocateIntBuffers(&intClip, &intSubj, clipSize, subjSize);
		
	result ? printf("success.\n"): printf("fail\n");
	printf("CopyData subj-----");
	
	if( result ) 
		result = CopyData(devSubj, subj, subjSize * sizeof(vertex), cudaMemcpyHostToDevice);
	
	result ? printf("success.\n"): printf("fail\n");
	printf("CopyData clip-----");
	
	if( result )
		result = CopyData(devClip, clip, clipSize * sizeof(vertex), cudaMemcpyHostToDevice);

	result ? printf("success.\n"): printf("fail\n");
	printf("CalcIntersection-----");
	
	// Calulate the intersection points.
	if( result ) {
	
		CalcIntersections<<<1, 128>>>(devClip, clipSize, devSubj, subjSize,
								  devIntClip, devIntSubj, devSize);
		
		devResult = cudaMemcpy(size, devSize, 2 * sizeof(int), cudaMemcpyDeviceToHost);
		if( devResult != cudaSuccess ) {
			result = 0;
		}
	result ? printf("success.\n"): printf("fail\n");
	
	} else

	if( result ) 
		result = CopyData(intSubj, devIntSubj, clipSize * subjSize * sizeof(vertex), cudaMemcpyDeviceToHost);
	
	if( result ) 
		result = CopyData(intClip, devIntClip, clipSize * subjSize * sizeof(vertex), cudaMemcpyDeviceToHost);
	
	printf("BuildIntPoly-----");

	if( result ) {

		result = BuildIntPoly(*intPoly, *intPolySize, intSubj, size[SUBJ], intClip, size[CLIP]);
	}

	result ? printf("success.\n"): printf("fail\n");
	printf("FreeDevMem-----");
	
	result = FreeDevMem(&devClip, &devSubj, &devIntClip, &devIntSubj);
		
	result ? printf("success.\n"): printf("fail\n");

	return result;
}





// 
//	Calculate the union polygon
//
//
int gpuUnion(VERTEX *subj, int subjSize, VERTEX *clip, int clipSize,
				 VERTEX **intPoly, int *intPolySize)
{
	int		result = 1;
	vertex	*devSubj, *devClip, *devIntSubj, *devIntClip,
			*intClip, *intSubj;
	int		*devSize, size[2];
	cudaError_t	devResult;

		
	result = AllocateDevMem(devSize, devClip, devSubj, devIntClip, devIntSubj, clipSize, subjSize);

	if( result )
		result = AllocateIntBuffers(&intClip, &intSubj, clipSize, subjSize);
		
	if( result ) 
		result = CopyData(devSubj, subj, subjSize * sizeof(vertex), cudaMemcpyHostToDevice);
	
	if( result )
		result = CopyData(devClip, clip, clipSize * sizeof(vertex), cudaMemcpyHostToDevice);


	// Calulate the intersection points.
	if( result ) {
	
		CalcIntersections<<<1, 128>>>(devClip, clipSize, devSubj, subjSize,
								  devIntClip, devIntSubj, devSize);
		
		devResult = cudaMemcpy(size, devSize, 2 * sizeof(int), cudaMemcpyDeviceToHost);
		if( devResult != cudaSuccess ) {
			result = 0;
		}
	} else

	if( result ) 
		result = CopyData(intSubj, devIntSubj, clipSize * subjSize * sizeof(vertex), cudaMemcpyDeviceToHost);
	
	if( result ) 
		result = CopyData(intClip, devIntClip, clipSize * subjSize * sizeof(vertex), cudaMemcpyDeviceToHost);
	
	if( result ) {

		result = BuildUnionPoly(*intPoly, *intPolySize, intSubj, size[SUBJ], intClip, size[CLIP]);
	}
	result = FreeDevMem(&devClip, &devSubj, &devIntClip, &devIntSubj);
		
	return result;
}








//
//
//	 C wrappers for the Postgres plugin
//
//
/*
extern "C" {


int gi_Intersect(VERTEX *poly1, int poly1Size, VERTEX *poly2, int poly2Size,
				   VERTEX **intPoly, int *intPolySize) 
{
	return gpuIntersect(poly1, poly1Size, poly2, poly2Size,
				 		intPoly, intPolySize);
}



int gi_Union(VERTEX *poly1, int poly1Size, VERTEX *poly2, int poly2Size,
				   VERTEX **intPoly, int *intPolySize) 
{
	return gpuIntersect(poly1, poly1Size, poly2, poly2Size,
				 		intPoly, intPolySize);
}

}
*/
