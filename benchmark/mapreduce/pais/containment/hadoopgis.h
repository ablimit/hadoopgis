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
//#include "IndexParam.h"


using namespace std;
using boost::lexical_cast;
using boost::bad_lexical_cast;

//using namespace SpatialIndex;

typedef boost::geometry::model::d2::point_xy<int> point;
typedef boost::geometry::model::polygon<point> polygon;

typedef map<string,map<int,vector<polygon> > > polygonmap;


const string bar= "|";
const string tab = "\t";
const char comma = ',';

const string shapebegin = "POLYGON((";
const string shapeend = "))";

