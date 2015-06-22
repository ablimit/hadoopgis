/**
* Returns the area in cartesian units. Area is negative if ring is oriented CCW, 
* positive if it is oriented CW and zero if the ring is degenerate or flat.
* http://en.wikipedia.org/wiki/Shoelace_formula
*/

export uniform double ptarray_signed_area(uniform double pa[] , const uniform int npoints)
{
	if (! pa || npoints < 3 )
		return 0.0;

	varying double sum = 0.0;
	//assert(programCount <= 8);

	for(int i= 0; i< npoints ; i+= programCount ) {
		sum += pa[i+programIndex] * pa[i+programIndex+3];
		sum -= pa[i+programIndex+1] * pa[i+programIndex+2];
	}

	return reduce_add(sum) * 0.5;	
}

