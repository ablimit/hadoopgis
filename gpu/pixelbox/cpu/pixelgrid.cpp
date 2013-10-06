#include <stdlib.h>
#include <stdio.h>
#include "tbb/parallel_reduce.h"
#include "tbb/parallel_for.h"
#include "tbb/blocked_range.h"
//#define IN_GPU_SPATIAL_LIB
#include "cpu_spatial.h"

using namespace tbb;

#define task_granularity	10

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

class PolyClipper;

class PolyAreaReducer
{
	mbr_t &mbr;
	int *x1, *y1, *x2, *y2;
	int start1, end1, start2, end2;

public:
	int area_inter, area_union;

	PolyAreaReducer(mbr_t &the_mbr,
					int *the_x1, int *the_y1, int the_start1, int the_end1,
					int *the_x2, int *the_y2, int the_start2, int the_end2) :
		mbr(the_mbr), x1(the_x1), y1(the_y1), start1(the_start1), end1(the_end1),
		x2(the_x2), y2(the_y2), start2(the_start2), end2(the_end2), area_inter(0), area_union(0) {}

	PolyAreaReducer(PolyAreaReducer& x, split) :
		mbr(x.mbr), x1(x.x1), y1(x.y1), start1(x.start1), end1(x.end1),
		x2(x.x2), y2(x.y2), start2(x.start2), end2(x.end2), area_inter(0), area_union(0) {}

	void operator()( const blocked_range<int>& r )
	{
		int start, end, inp1, inp2;

		for(int i = r.begin(); i != r.end(); i++) {
			start = i * task_granularity;
			end = start + task_granularity;

			for(int ipoint = start; ipoint < end; ipoint++)
			{
				int x = ipoint % (mbr.r - mbr.l + 1) + mbr.l;
				int y = ipoint / (mbr.t - mbr.b + 1) + mbr.b;

				inp1 = in_poly(x, y, x1, y1, start1, end1);	// in poly1?
				inp2 = in_poly(x, y, x2, y2, start2, end2);	// in poly2?
				area_inter += inp1 & inp2;
				area_union += inp1 | inp2;
			}
		}
	}

	void join( const PolyAreaReducer& y )
	{
		area_inter += y.area_inter;
		area_union += y.area_union;
	}
};

class PolyClipper
{
	float *ratios;
	mbr_t *mbrs;
	int *idx1, *idx2;
	int *offsets1, *offsets2;
	int *x1, *y1, *x2, *y2;

public:
	PolyClipper(float *the_ratios, mbr_t *the_mbrs,
				int *the_idx1, int *the_idx2,
				int *the_offsets1, int *the_offsets2,
				int *the_x1, int *the_y1, int *the_x2, int *the_y2)
		: ratios(the_ratios), mbrs(the_mbrs),
		idx1(the_idx1), idx2(the_idx2), offsets1(the_offsets1), offsets2(the_offsets2),
		x1(the_x1), y1(the_y1), x2(the_x2), y2(the_y2)
	{}

	void operator()(const blocked_range<int>& r) const
	{
		float *r1 = ratios;

		for(int i = r.begin(); i != r.end(); i++) {
			int nr_pixels = (mbrs[i].r - mbrs[i].l + 1) * (mbrs[i].t - mbrs[i].b + 1);
			int nr_tasks = nr_pixels / task_granularity + 1;

			PolyAreaReducer areas(mbrs[i], x1, y1, offsets1[idx1[i]], offsets1[idx1[i]+1],
				x2, y2, offsets2[idx2[i]], offsets2[idx2[i]+1]);

			parallel_reduce(blocked_range<int>(0, nr_tasks), areas);

			r1[i] = areas.area_inter / (float)areas.area_union;
		}
	}
};

float *cpu_clip(
	int nr_poly_pairs, mbr_t *mbrs,	// mbr of each poly pair
	int *idx1, int *idx2,			// index to poly_array 1 and 2
	int no1, int *offsets1,			// offset to poly_arr1's vertices
	int no2, int *offsets2,			// offset to poly_arr2's vertices
	int nv1, int *x1, int *y1,// poly_arr1's vertices
	int nv2, int *x2, int *y2)// poly_arr2's vertices
{
	float *ratios = NULL;

	ratios = (float *)malloc(nr_poly_pairs * sizeof(float));
	if(!ratios) {
		perror("malloc error for ratios\n");
		return NULL;
	}

	parallel_for(blocked_range<int>(0, nr_poly_pairs),
		PolyClipper(ratios, mbrs, idx1, idx2, offsets1, offsets2, x1, y1, x2, y2));

	return ratios;
}
