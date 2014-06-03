#ifndef SPATIAL_H
#define SPATIAL_H

#define A_VERY_LARGE_NUM	1073741824
#define A_VERY_SMALL_NUM	(-1073741824)

#define FREE(m)	if(m) free(m)

#define MIN(x, y)	((x) < (y) ? (x) : (y))
#define MAX(x, y)	((x) > (y) ? (x) : (y))

#define DIFF_TIME(t1, t2) ( \
    ((t2).tv_sec + ((double)(t2).tv_usec)/1000000.0) - \
	((t1).tv_sec + ((double)(t1).tv_usec)/1000000.0) \
)


// minimum bounding rectangular
typedef struct mbr_struct
{
	int l, r, b, t;		// left < right, bottom < top
} mbr_t;

// polygons array
typedef struct poly_array_struct
{
	int nr_polys;		// number of polygons in this array
	mbr_t *mbrs;		// mbrs
	int *offsets;		// offset of each poly in x and y arrays
	int nr_vertices;	// number of vertices for all polys in this array
	int *x;			// x coordinates
	int *y;			// y coordinates
} poly_array_t;

// polygon pairs array: indexes into the polygon arrays
// are stored, instead of the actual polygon data
typedef struct poly_pair_array_struct
{
	int nr_poly_pairs;	// number of polygon pairs
	mbr_t *mbrs;		// mbrs of each polygon pair
	int *idx1;			// index to the first of each polygon pair
	int *idx2;			// index to the second of each polygon pair
} poly_pair_array_t;


inline int mbr_area(const mbr_t *mbr);
inline int mbr_area_2(const mbr_t *mbr1, const mbr_t *mbr2);
inline int mbr_area_inc(const mbr_t *mbr, const mbr_t *mbr_inc);
inline int mbr_diff(const mbr_t *mbr1, const mbr_t *mbr2);
inline int in_mbr(const mbr_t *mbr, int x, int y);
inline int mbr_intersect(const mbr_t *mbr1, const mbr_t *mbr2);
inline int mbr_update(mbr_t *mbr, const mbr_t *mbr_inc);
inline void mbr_merge(mbr_t *merge_to, const mbr_t *mbr1, const mbr_t *mbr2);
void init_poly_array(poly_array_t *polys);
void fini_poly_array(poly_array_t *polys);
void init_poly_pair_array(poly_pair_array_t *poly_pairs);
int make_poly_pair_array(poly_pair_array_t *poly_pairs, const int nr_poly_pairs);
void fini_poly_pair_array(poly_pair_array_t *poly_pairs);
void free_poly_pair_array(poly_pair_array_t *poly_pairs);

#endif
