/**
 * Returns the area in cartesian units. Area is negative if ring is oriented CCW, 
 * positive if it is oriented CW and zero if the ring is degenerate or flat.
 * http://en.wikipedia.org/wiki/Shoelace_formula
 */

// utility function to read the value of programCount from C/C++
export uniform int32 get_programCount() {
	return programCount;
}

export uniform float ptarray_signed_area_aos(uniform float pa[] , const uniform int npoints)
{
	if (! pa || npoints < 3 )
		return 0.0;

	uniform float * uniform ap = pa;
	varying float sum = 0.0;
	//assert(programCount <= 8);
	//int step = programCount*2; 

	//for(int i= programIndex; i< npoints ; i+=step) {

	foreach(i=0 ... npoints/2) {
		sum += pa[i*2] * pa[i*2+3] - pa[i*2+1] * pa[i*2+2];
	}

	return reduce_add(sum) * 0.5;	
}
/*
export uniform float ptarray_signed_area_soa(uniform float pa[], const uniform int npoints)
{
	if (! pa || npoints < 3 )
		return 0.0;

	assert(programCount <= 16);

	uniform float * uniform ap = pa;
	varying float sum = 0.0;
	uniform int iter = npoints/programCount;


	for (uniform int i = 0; i < iter; i++, ap += 16) {
		for (uniform int j = 0; j < 16; j += programCount) {
			float x0 = ap[j + programIndex];
			float y0 = ap[8 + j + programIndex];
			float x1 = ap[j +1 + programIndex];
			float y1 = ap[8 + j + 1 + programIndex];
			sum += (x0*y1 - x1*y0); 
		}
	}
	
	iter = (npoints %16)/2; 
	uniform int dist = iter+1;

	for (uniform int j = 0; j <iter ; j += programCount) {
		if (programIndex>=iter)
			continue;
		float x0 = ap[j + programIndex];
		float x1 = ap[j +1 + programIndex];
		float y0 = ap[dist + j + programIndex];
		float y1 = ap[dist + j + 1 + programIndex];
		//results[j] = x0*y1;
		//results[j+1] = x1*y0); 
		sum += (x0*y1-x1*y0); 
	}

	return reduce_add(sum) * 0.5;	
}
*/

export uniform float ptarray_signed_area_soa(uniform float pa[] , const uniform int npoints)
{
	if (! pa || npoints < 3 )
		return 0.0;

	uniform float * uniform ap = pa;
	varying float sum = 0.0;
	assert(programCount <= 8);
	foreach (i = 0 ... npoints) {
	//for (uniform int i = 0; i <npoints ; i+= programCount) {
		float x0,y0,x1,y1;
		//varying int base = 2*i + programIndex;
		aos_to_soa4(&pa[2*i], &x0, &y0, &x1, &y1);
		sum += x0*y1 - x1*y0;
	}

	return reduce_add(sum) * 0.5;	
}

