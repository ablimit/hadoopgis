#include <stdlib.h>
#include <stdio.h>
#include "../cpu_spatial.h"

// compute whether point (x, y) lays within a polygon whose contour
// is specified by xs and ys in the range of [istart, iend); the last
// vertex repeats the first vertex
static int in_poly(
    const int x, const int y,
    const int *xs, const int *ys,
    const int start, const int end)
{
  // Count the number of intersections between the horizontal beam from (x,y) to right infinity
  // and the edges of the polygon. Odd number indicates the point lies within the polygon.
  int crossing = 0;
  for(int i = start; i < end-1; i++)
  {
    int startx = xs[i], starty = ys[i];
    int endx = xs[i+1], endy = ys[i+1];

    // A horizontal edge? Then no crossing.
    if(endy == starty)
    {
      continue;
    }

    // If this is an upword edge
    if(endy > starty)
    {
      if(y == starty && x <= startx) {
        crossing++;
      }
      /*			if(y == starty && x <= startx)
              {
              crossing++;
              }
              else if(y >= starty && y < endy)
              {
              if(x < startx + (y - starty) * (endx - startx) * 1.0 / (endy - starty))
              {
              crossing++;
              }
              }
              */		}
      // If this is a downward edge
    else
    {
      if(y == endy && x <= endx) {
        crossing++;
      }
      /*			if(y == endy && x <= endx)
              {
              crossing++;
              }
              else if(y >= endy && y < starty)
              {
              if(x < startx + (starty - y) * (endx - startx) * 1.0 / (starty - endy))
              {
              crossing++;
              }
              }
              */		}
  }

  return crossing & 1;
}

float *clip(
    int stream_no,
    const int nr_poly_pairs, // mbr of each poly pair
    const mbr_t *mbrs,
    const int *idx1, const int *idx2,			// index to poly_array 1 and 2
    const int no1, const int *offsets1,			// offset to poly_arr1's vertices
    const int no2, const int *offsets2,			// offset to poly_arr2's vertices
    const int nv1, const int *x1, const int *y1,// poly_arr1's vertices
    const int nv2, const int *x2, const int *y2)// poly_arr2's vertices
{
  float *ratios = NULL;

  ratios = (float *)malloc(nr_poly_pairs * sizeof(float));
  if(!ratios) {
    perror("malloc error for ratios\n");
    return NULL;
  }

  for (int i = 0 ; i< nr_poly_pairs ; i++){
    int area_union = 0 ; 
    int area_inter = 0 ; 
    int inp1, inp2, x, y;

    size_t nr_pixels = (size_t)(mbrs[i].r - mbrs[i].l + 1) * (mbrs[i].t - mbrs[i].b + 1);

    for(size_t ipoint = 0; ipoint < nr_pixels; ipoint++)
    {
      x = ipoint % (mbrs[i].r - mbrs[i].l + 1) + mbrs[i].l;
      y = ipoint / (mbrs[i].t - mbrs[i].b + 1) + mbrs[i].b;

      inp1 = in_poly(x, y, x1, y1, offsets1[idx1[i]], offsets1[idx1[i]+1]);	// in poly1?
      inp2 = in_poly(x, y, x2, y2, offsets2[idx2[i]], offsets2[idx2[i]+1]);	// in poly2?
      area_inter += inp1 & inp2;
      area_union += inp1 | inp2;
    }

    ratios[i] = area_inter / (float)area_union;
  }

  return ratios;
}
