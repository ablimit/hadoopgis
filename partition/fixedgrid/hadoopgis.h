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

#define FillFactor 0.7
#define IndexCapacity 100
#define LeafCapacity 50
#define COMPRESS true
#define TILE_SIZE 4096


using namespace std;
using namespace SpatialIndex;


const string bar= "|";
const string tab = "\t";
const string comma = ",";
const string space = " ";

const string shapebegin = "POLYGON((";
const string shapeend = "))";

