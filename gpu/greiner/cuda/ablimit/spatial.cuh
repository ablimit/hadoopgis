#if !defined(_SPATIAL_)
#define _SPATIAL_

#define COOR float

typedef struct vertex {
	COOR	x;
	COOR	y;
	float	alpha;
	int		next;
	bool	internal;
	int		linkTag;
} vertex;


#endif
