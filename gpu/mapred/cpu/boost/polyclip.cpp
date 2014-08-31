#include "cpu_refine.h"

#include <boost/foreach.hpp>
#include <boost/geometry.hpp>
#include <boost/geometry/geometry.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometries/polygon.hpp>

using namespace std;
//using namespace boost;

// data type declaration 
typedef boost::geometry::model::d2::point_xy<int> point;
typedef boost::geometry::model::polygon<point> polygon;


float * refine(
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
    float area_union = 0.0 ; 
    float area_inter = 0.0 ; 
    polygon p; polygon q;

    for (int j = offsets1[idx1[i]]; j < offsets1[idx1[i]+1] - 1; j++ )
      p.outer().push_back(point(x1[j], y1[j]));
    boost::geometry::correct(p);
    for (int j = offsets2[idx2[i]]; j < offsets2[idx2[i]+1] - 1; j++ )
      q.outer().push_back(point(x2[j], y2[j]));
    boost::geometry::correct(q);

    std::vector<polygon> output;
    boost::geometry::intersection(p,q, output);
    if (output.size()==0) 
    {
      //std::cerr << "0" << std::endl;
      ratios[i] = 0.0; 
      continue ;
    }
    BOOST_FOREACH(polygon const& a, output)
    {
      area_inter += boost::geometry::area(a) ;
    }

    area_union = boost::geometry::area(p) + boost::geometry::area(q) - area_inter ; 
    ratios[i] = area_inter / area_union;
  }

  return ratios;
}
