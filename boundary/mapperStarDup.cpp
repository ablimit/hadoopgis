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
const string tab = "\t";
const char comma = ',';
const string doublequote = "\"";
const int tileSize = 2048; /* Default value */
const char minusS = '-';

const string shapebegin = "POLYGON((";
const string shapeend = "))";


int getJoinIndex ()
{
    char * filename = getenv("map_input_file");
    //char * filename = "astroII.1.1";
    if ( NULL == filename ){
	   cerr << "map.input.file is NULL." << endl;
	   return 1 ;
    }
    int len= strlen(filename);
    //int len= filename.size();
    int index = filename[len-1] - '0' ;
    return index;
}

int main(int argc, char **argv) {
    int index = getJoinIndex();
    if (index < 1)
    {
		cerr << "InputFileName index is corrupted.. " << endl;
		return 1; //failure 
    }
	//int index = 1;

    string input_line;
    string key ;
    string value;
	string imgid;	
	string polyid;
	size_t pos;
	size_t pos2;
	int tmpX;
	int tmpY;
	int maxX;
	int maxY;
	int minX;
	int minY;
	int i, j, k; /* Loop counter */

	polygon * shape= NULL;
    box * mbb = NULL;

    while(cin && getline(cin, input_line) && !cin.eof()){
	
	pos2 = input_line.find_first_of(minusS,0);	
	imgid = input_line.substr(0,pos2); /* Name of the image */

	/* pos becomes the comma in front of the */
	pos = input_line.find_first_of(comma,0);

	if (pos == string::npos)
	    return 1; // failure

	pos = input_line.find_first_of(comma,pos+1);

	/* Retrieve the 7-digit polygon ID*/
	polyid = input_line.substr(pos - 8, 7);

	/* Polygon markup */
	value= input_line.substr(pos + 2, input_line.size() - pos - 3);	

	shape = new polygon();
	mbb   = new box();

	boost::geometry::read_wkt(shapebegin+value+shapeend,*shape);
	boost::geometry::correct(*shape);
	boost::geometry::envelope(*shape, *mbb);

	minX =  boost::geometry::get<boost::geometry::min_corner, 0>(*mbb);
	minY =  boost::geometry::get<boost::geometry::min_corner, 1>(*mbb);

	minX = ((int) ( minX / tileSize )) * tileSize;
	minY = ((int) ( minY / tileSize )) * tileSize;

	maxX = boost::geometry::get<boost::geometry::max_corner, 0>(*mbb);
	maxY = boost::geometry::get<boost::geometry::max_corner, 1>(*mbb);

	tmpX = (maxX - minX) / tileSize + 1;
	tmpY = (maxY - minY) / tileSize + 1;

	for (i = 0; i < tmpX; i++) {
		for (j = 0; j < tmpY; j++) {
			/*  key is imgid-minX-minY */
			cout << imgid << minusS << (minX + i * tileSize) << (minY + j * tileSize) 
			<< tab << index 
			<< tab << polyid 
			<< tab << value << endl;

			/* An alternate tile-format with fixed size
				cout << imgid << minusS 
			<< setfill('0') << setw(10) << (minX + i * tileSize) 
			<< minusS 
		    << setfill('0') << setw(10) << (minY + j * tileSize) 
			<< tab << index 
			<< tab << polyid 
			<< tab << value << endl;
			*/

		}
	}
    }
    return 0; // success
}

