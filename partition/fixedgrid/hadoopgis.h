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
#include <spatialindex/SpatialIndex.h>

// geos 
#include <geos/geom/PrecisionModel.h>
#include <geos/geom/GeometryFactory.h>
#include <geos/geom/Geometry.h>
#include <geos/geom/Point.h>
#include <geos/io/WKTReader.h>


#define FillFactor 0.7
#define IndexCapacity 100
#define LeafCapacity 50
#define COMPRESS true
#define TILE_SIZE 4096


using namespace std;
using namespace SpatialIndex;

using namespace geos;
using namespace geos::io;
using namespace geos::geom;

// using boost::lexical_cast;
// using boost::bad_lexical_cast;



const string BAR = "|";
const string TAB = "\t";
const string COMMA = ",";
const string SPACE = " ";

const string shapebegin = "POLYGON((";
const string shapeend = "))";

