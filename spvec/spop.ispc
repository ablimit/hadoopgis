/**
* Returns the area in cartesian units. Area is negative if ring is oriented CCW, 
* positive if it is oriented CW and zero if the ring is degenerate or flat.
* http://en.wikipedia.org/wiki/Shoelace_formula
*/

// utility function to read the value of programCount from C/C++
export uniform int32 get_programCount() {
  return programCount;
}

export uniform double ptarray_signed_area_aos(uniform double pa[] , const uniform int npoints)
{
	if (! pa || npoints < 3 )
		return 0.0;

  uniform double * uniform ap = pa;
	varying double sum = 0.0;
	//assert(programCount <= 8);
  //uniform int step = programCount*2; 

	//for(int i= programIndex; i< npoints ; i+=step) {
	foreach(i=0 ... npoints/2) {
		sum += pa[i*2] * pa[i*2+3] - pa[i*2+1] * pa[i*2+2];
	}

	return reduce_add(sum) * 0.5;	
}

export uniform double ptarray_signed_area_soa(uniform double pa[] , const uniform int npoints)
{
  if (! pa || npoints < 3 )
    return 0.0;

  assert(programCount <= 8);

  uniform double * uniform ap = pa;
  varying double sum = 0.0;
  uniform int iter = npoints/16;


  for (uniform int i = 0; i < iter; i++, ap += 16) {
    for (uniform int j = 0; j < 8; j += programCount) {
      double x0 = ap[j + programIndex];
      double y0 = ap[8 + j + programIndex];
      double x1 = ap[j +1 + programIndex];
      double y1 = ap[8 + j + 1 + programIndex];
      sum += (x0*y1 - x1*y0); 
    }
  }

  iter = (npoints %16)/2; 
  uniform int dist = iter+1;

  for (uniform int j = 0; j <4 ; j += programCount) {
    double x0 = ap[j + programIndex];
    double x1 = ap[j +1 + programIndex];
    double y0 = ap[6+ j + programIndex];
    double y1 = ap[7+ j + programIndex];
    //results[j] = x0*y1;
    //results[j+1] = x1*y0); 
    sum += (x0*y1-x1*y0); 
  }

  return reduce_add(sum) * 0.5;	
}

/*
   export uniform double ptarray_signed_area(uniform struct Point uniform pa[] , const uniform int npoints)
   {
   if (! pa || npoints < 3 )
   return 0.0;

//uniform double * uniform ap = pa;
varying double sum = 0.0;
assert(programCount <= 8);

foreach (i = 0 ... npoints) {
sum += pa[i].x * pa[i+1].y;
sum -= pa[i].y * pa[i+1].x;
}

return reduce_add(sum) * 0.5;	
}
 */
