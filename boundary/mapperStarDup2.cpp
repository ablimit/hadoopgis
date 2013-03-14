#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <cmath>
#include <fstream>
#include <iomanip>

#include <boost/foreach.hpp>
#include <boost/geometry.hpp>
#include <boost/geometry/geometry.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometries/polygon.hpp>
#include <boost/geometry/geometries/box.hpp>
#include <boost/geometry/domains/gis/io/wkt/wkt.hpp>
#include <boost/geometry/domains/gis/io/wkt/wkt.hpp>

using namespace std;

typedef boost::geometry::model::d2::point_xy<int> point;
typedef boost::geometry::model::polygon<point> polygon;
typedef boost::geometry::model::box<point> box;

typedef map<string,map<int,vector<polygon*> > > polymap;
typedef map<string,map<int,vector<box*> > > boxmap;

const char offset = '1';
const char tab = '\t';
const char comma = ',';
const int tileSize = 2048; /* Default tile size */
const char minusS = '-';

/* The polygon string format for C++ */
const string shapebegin = "POLYGON((";
const string shapeend = "))";

int getJoinIndex();

int getJoinIndex ()
{
    char * filename = getenv("map_input_file");

    if ( NULL == filename ){
	   cerr << "map.input.file is NULL." << endl;
	   return 1 ;
    }
    int len= strlen(filename);
    int index = filename[len-1] - '0';
    return index;
}

int main(int argc, char **argv) {
	/* Read the join index from the filename */
    int index = getJoinIndex();
    if (index < 1)
    {
		cerr << "InputFileName index is corrupted.. " << endl;
		return 1; //failure 
    }

    string input_line;
    string value;
	string imgid;	
	string polyid;
	string polystring; /* The string containing all vertices in a comma delimited format */
	size_t pos, pos2; /* Index position in string */

	int tmpX, tmpY, maxX, maxY, minX, minY; /* Temporary coordinates  */
	int i, j, k; /* Loop counter */
	polygon * shape; /* Contain the temporary polygon */
    box * mbb; /* The minimum bounding rectangle */

	shape = NULL;
	mbb = NULL;

    while(cin && getline(cin, input_line) && !cin.eof()) {
		pos2 = input_line.find_first_of(minusS,0);	
		imgid = input_line.substr(0,pos2); /* Name of the image */

		/* pos becomes the comma in front of the */
		pos2 = input_line.find_first_of(comma,0);

		if (pos == string::npos)
			return 1; // failure

		pos = input_line.find_first_of(comma,pos2+1);

		/* Retrieve the 10-digit polygon ID*/
		polyid = input_line.substr(pos2+1,pos-pos2-1);
        //cerr << polyid << endl;

		/* Polygon markup */
		value= input_line.substr(pos + 2, input_line.size() - pos - 3);	

		polystring = shapebegin+value+shapeend;
	
		/* Create a new empty polygon */
		shape = new polygon();
		mbb   = new box();

		/* Read the polygon markup and minimum bounding box */
		boost::geometry::read_wkt(polystring,*shape);
		boost::geometry::correct(*shape);
		boost::geometry::envelope(*shape, *mbb);

		/* Obtain the coordinates of lower-left-most corner */
		minX =  boost::geometry::get<boost::geometry::min_corner, 0>(*mbb);
		minY =  boost::geometry::get<boost::geometry::min_corner, 1>(*mbb);

		maxX = boost::geometry::get<boost::geometry::max_corner, 0>(*mbb);
		maxY = boost::geometry::get<boost::geometry::max_corner, 1>(*mbb);

		minX = ((int) ( minX / tileSize )) * tileSize;
		minY = ((int) ( minY / tileSize )) * tileSize;

		tmpX = (maxX - minX) / tileSize + 1;
		tmpY = (maxY - minY) / tileSize + 1;

		/* Emit the polygon to appropriate tiles */
		for (i = 0; i < tmpX; i++) {
			for (j = 0; j < tmpY; j++) {
				/*  key is tileid == imgid-minX-minY */
				cout << imgid << minusS << (minX + i * tileSize) << minusS << (minY + j * tileSize) 
				<< tab << index << tab << polyid << tab << polystring << endl;
			}
		}
    }
    return 0; // success
}

