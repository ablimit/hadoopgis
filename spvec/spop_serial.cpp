/**
* Returns the area in cartesian units. Area is negative if ring is oriented CCW, 
* positive if it is oriented CW and zero if the ring is degenerate or flat.
* http://en.wikipedia.org/wiki/Shoelace_formula
*/

double ptarray_signed_area(double pa[] , const int npoints)
{
	if (! pa || npoints < 3 )
		return 0.0;

	double sum = 0.0;
	//assert(programCount <= 8);

	for(int i= 0 ; i< npoints; i+=2) {
		sum += (pa[i] * pa[i+3] - pa[i+1] *pa[i+2]);
	}

	return sum * 0.5;	
}


