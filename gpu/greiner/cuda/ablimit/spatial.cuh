#if !defined(_SPATIAL_)
#define _SPATIAL_

#define point_type float

typedef struct vertex {
	point_type	x;
	point_type	y;
	float	alpha;
	int		next;
	bool	internal;
	int		linkTag;
} vertex;


#endif
