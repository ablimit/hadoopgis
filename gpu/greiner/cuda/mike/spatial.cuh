#if !defined(_SPATIAL_)
#define _SPATIAL_

typedef struct vertex {
	float	x;
	float	y;
	float	alpha;
	int		next;
	bool	internal;
	int		linkTag;
}VERTEX;


#endif
