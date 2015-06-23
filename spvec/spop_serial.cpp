/**
* Returns the area in cartesian units. Area is negative if ring is oriented CCW, 
* positive if it is oriented CW and zero if the ring is degenerate or flat.
* http://en.wikipedia.org/wiki/Shoelace_formula
*/
#include<stdio.h>

double ptarray_signed_area_aos(double pa[] , const int npoints)
{
	if (! pa || npoints < 3 )
		return 0.0;

	double sum = 0.0;
	//assert(programCount <= 8);

	for(int i= 0 ; i< npoints; i+=2) {
		sum += (pa[i] * pa[i+3] - pa[i+1] *pa[i+2]);
    //printf("coef %d : %f\n", i, a);
	}

	return sum * 0.5;	
}


double ptarray_signed_area_soa(double pa[] , const int npoints)
{
	if (! pa || npoints < 3 )
		return 0.0;

  double sum = 0.0;
  double * xp = pa;
  
  for(int i= 0 ; i<npoints/16; i++, xp+=16) {
    for(int j= 0 ; j<8; j++) {
      //printf("+ %f : -%f\n", a, b);
      sum += (xp[j] * xp[j+9] - xp[j+1] *xp[j+8]);
    }
  }

  int remain = (npoints % 16)/2; 
  //printf("remain %d\n", remain);
  for(int j= 0 ; j<remain; j++) {
    //printf("+ %f : -%f\n", a, b);
    sum += (xp[j] * xp[j+remain+2] - xp[j+1] *xp[j+remain+1]);
  }
  //printf("sum: %f\n", sum);
  return sum * 0.5;	
}

