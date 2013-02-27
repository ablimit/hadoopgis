#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <map>
#include <string>
#include <cmath>

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


int tileSize = 2048;
int MULTI = 1;

const char minusS = '-';

const string shapebegin = "POLYGON((";
const string shapeend = "))";

int main(int argc, char **argv) {
	if (argc != 4) {
		cerr << "Need argument: tilesize multiple-dup";
	}
	tileSize = atoi(argv[1]);
	MULTI = atoi(argv[2]);

    string tab = "\t";
    char comma = ',';
    string input_line;
    string key ;
    string value;
	
	string imgid;

	size_t pos;
	size_t pos2;
	int tmpX;
	int tmpY;
	int maxX;
	int maxY;
	int minX;
	int minY;
	int i, j; /* Loop counter */
	int numObjects;
	int dup;

	string polyid;
	string newpolyid;

	polygon * shape= NULL;
    box * mbb = NULL;

	numObjects = 0;
	dup = 0;

    while(cin && getline(cin, input_line) && !cin.eof()){
	pos2 = input_line.find_first_of(minusS,0);	
	imgid = input_line.substr(0,pos2); /* Name of the image */

	/* pos becomes the comma in front of the */
	pos = input_line.find_first_of(comma,0);

	
	if (pos == string::npos)
	    return 1; // failure

	// key = input_line.substr(0,pos);

	pos = input_line.find_first_of(comma,pos+1);

	/* Retrieve the 6 digit polygon ID*/
	polyid = input_line.substr(pos - 8, 7);

	/* Ignore double quotes */
	value= input_line.substr(pos + 2, input_line.size() - pos - 3);	
	// Read spatial input 

	shape = new polygon();
	mbb   = new box();	
	// cout << "value is " << value << endl ;

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

	// 4 coordinates
	// cout << minX << " " << minY << " and " << maxX << " " << maxY << endl;

	if (tmpX == 1 && tmpY == 1) {
		cout << input_line << endl; // not a boundary object
		numObjects++;
	} else {
		/* Generate multiple boundary objects */
		for (i = 1; i <= MULTI; i++) {
			cout << imgid << minusS << setfill('0') << setw(10) << minX
				<< minusS				
				<< setfill('0') << setw(10) << minY
				<< ",+" << setfill('0') << setw(29) << polyid;
			// cout << i << ".," << doublequote<<  value << doublequote << " dup! "<< endl; 
			cout << i << ".," << doublequote<<  value << doublequote << endl; 
			numObjects++;			
			dup++;
		}
	}
	
    }

	ofstream statfile(argv[3]);
	if (statfile.is_open()) {
		statfile << dup << "/" << numObjects << " " << ((double) dup) / numObjects << endl;
		statfile.close();		
	}

    return 0; // success
}

