#include <math.h>
#include <dirent.h>
#include <unistd.h>
#include <errno.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <iostream>
#include <string.h>
#include <string>

#include <vector>
#include <map>

#include <boost/foreach.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/geometry.hpp>
#include <boost/geometry/geometry.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometries/polygon.hpp>
#include <boost/geometry/geometries/box.hpp>
#include <boost/geometry/domains/gis/io/wkt/wkt.hpp>

#include <spatialindex/SpatialIndex.h>

#define FillFactor 0.7
#define IndexCapacity 100
#define LeafCapacity 50
#define COMPRESS true


using namespace std;
using namespace SpatialIndex;

using boost::lexical_cast;
using boost::bad_lexical_cast;

typedef boost::geometry::model::d2::point_xy<int> point;
typedef boost::geometry::model::polygon<point> polygon;
typedef boost::geometry::model::box<point> box;

typedef map<string,map<int,vector<polygon> > > polygonmap;


const string bar= "|";
const string tab = "\t";
const char comma = ',';

const string shapebegin = "POLYGON((";
const string shapeend = "))";


