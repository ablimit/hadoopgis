/**
* Returns the area in cartesian units. Area is negative if ring is oriented CCW, 
* positive if it is oriented CW and zero if the ring is degenerate or flat.
* http://en.wikipedia.org/wiki/Shoelace_formula
*/
#include<stdio.h>

float ptarray_signed_area_aos(float pa[] , const int npoints)
{
	if (! pa || npoints < 3 )
		return 0.0;

	float sum = 0.0;
	//assert(programCount <= 8);

	for(int i= 0 ; i< npoints; i+=2) {
			float x0 = pa[i];
			float x1 = pa[i+2];
			float y0 = pa[i + 1];
			float y1 = pa[i+3];
			sum += (x0*y1 - x1*y0); 
      //printf("%f : %f\n", x0*y1, x1*y0);
		//sum += (pa[i] * pa[i+3] - pa[i+1] *pa[i+2]);
    //printf("coef %d : %f\n", i, a);
	}

	return sum * 0.5;	
}


float ptarray_signed_area_soa(float pa[] , const int npoints)
{
	if (! pa || npoints < 3 )
		return 0.0;

  float sum = 0.0;
  float * xp = pa;
  float * np = xp+16;
  int iter = npoints/16; 
  int remain = (npoints % 16)/2;
  int dist = remain +1 ;
  
  for(int i= 0 ; i<iter; i++, xp+=16, np+=16) {
    for(int j= 0 ; j<8; j++) {
			float x0 = xp[j];
			float x1 = j==7 ? np[0] : xp[j + 1];
			float y0 = xp[8 + j ];
			float y1 = j==7 ? np[i+1< iter ? 8 : dist] : xp[8 + j + 1];
			sum += (x0*y1 - x1*y0); 
      //printf("%f : %f\n", x0*y1, x1*y0);
      //sum += (xp[j] * xp[j+9] - xp[j+1] *xp[j+8]);
    }
  }

  for(int j= 0 ; j<remain; j++) {
    //printf("+ %f : -%f\n", a, b);
    sum += (xp[j] * xp[j+dist+1] - xp[j+1] *xp[j+dist]);
  }
  //printf("sum: %f\n", sum);
  return sum * 0.5;	
}

