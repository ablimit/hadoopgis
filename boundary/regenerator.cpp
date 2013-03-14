#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <map>
#include <string>
#include <iterator>
#include <cmath>

#include <boost/foreach.hpp>
#include <boost/geometry.hpp>
#include <boost/geometry/geometry.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometries/polygon.hpp>
#include <boost/geometry/geometries/box.hpp>
#include <boost/geometry/domains/gis/io/wkt/wkt.hpp>
#include <boost/geometry/domains/gis/io/wkt/wkt.hpp>
#include <boost/algorithm/string.hpp>

using namespace std;

typedef boost::geometry::model::d2::point_xy<int> point;
typedef boost::geometry::model::polygon<point> polygon;
typedef boost::geometry::model::box<point> box;

typedef map<string,map<int,vector<polygon*> > > polymap;
typedef map<string,map<int,vector<box*> > > boxmap;

const char TAB = '\t';
const string COMMA = ",";
const char doublequote = '\"';
const string emptystring = "";
const string space = " ";
const string UNDERSCOR = "_";

int tileSize = 2048;
int MULTI = 1;

const char DASH = '-';
const string shapebegin = "POLYGON((";
const string shapeend = "))";

int main(int argc, char **argv) {
	if (argc != 4) {
		cerr << "Need argument: [tilesize] [multiple-dup] [statisticsfilesuffix]" <<endl;
        return 1;
	}
	tileSize = atoi(argv[1]);
	MULTI = atoi(argv[2]);

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
	int offsetDistance;
	int xCord;
	int yCord;

	string polyid;
	string newpolyid;
	string newvalue;
	vector<string> strs;
	vector<string> pairCords;
	vector<string>::const_iterator it; /* Iterator */

	string pairCoords;
    
	polygon * shape= NULL;
    box * mbb = NULL;

	numObjects = 0;
	dup = 0;
	offsetDistance = (int) (tileSize / MULTI);

    while(cin && getline(cin, input_line) && !cin.eof()){
	pos2 = input_line.find_first_of(DASH,0);	
	imgid = input_line.substr(0, pos2); /* Name of the image */

	/* pos becomes the COMMA in front of the */
	pos = input_line.find_first_of(COMMA,0);

	
	if (pos == string::npos)
	    return 1; // failure

	pos = input_line.find_first_of(COMMA,pos+1);

	/* Retrieve the 10 digit polygon ID*/
	polyid = input_line.substr(pos - 11, 10);

	/* Ignore double quotes */
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

	if (tmpX == 1 && tmpY == 1) {
			cout << imgid << DASH << setfill('0') << setw(10) << minX
				<< DASH				
				<< setfill('0') << setw(10) << minY
				<< COMMA << polyid << COMMA << doublequote <<value << doublequote <<endl;
		numObjects++;
	} else {
		/* Generate multiple boundary objects */
		boost::split(strs, value, boost::is_any_of(COMMA));

		for (i = 1; i <= MULTI; i++) {
			/* Splitting each pair */
			cout << imgid << DASH << setfill('0') << setw(10) << minX
				<< DASH				
				<< setfill('0') << setw(10) << minY
				<< COMMA << polyid << UNDERSCOR << i << COMMA << doublequote;

			if (tmpX > 1) {
				/* Span 2 tiles horizontally so we increment y-coordinates accordingly */
				for (it = strs.begin(); it != strs.end() - 1; ++it) {
					/* Splitting x-y coordinates */
					pairCords.clear();
					boost::split(pairCords, *it, boost::is_any_of(space));

					cout << pairCords.at(0) << space 
					<< ((boost::lexical_cast<int>(pairCords.at(1))) + i * offsetDistance) 
					<< COMMA << space;
					
				}

				/* Add the last pair */
				pairCords.clear();
				boost::split(pairCords, *it, boost::is_any_of(space));
				cout << pairCords.at(0) << space << ((boost::lexical_cast<int>(pairCords.at(1))) + i * offsetDistance);

			} else {
				/* Span 2 tiles vertically so we increment x-coordinates accordingly*/
				/* Span 2 tiles horizontally so we increment y-coordinates accordingly */
				for (it = strs.begin(); it != strs.end() - 1; ++it) {
					/* Splitting x-y coordinates */
					pairCords.clear();
					boost::split(pairCords, *it, boost::is_any_of(space));
					
					cout << (boost::lexical_cast<int>(pairCords.at(0)) + i * offsetDistance) 
					<< space << pairCords.at(1) << COMMA << space;
				}

				/* Add the last pair */
				pairCords.clear();
				boost::split(pairCords, *it, boost::is_any_of(space));
				cout << (boost::lexical_cast<int>(pairCords.at(0)) + i * offsetDistance) << space << pairCords.at(1);
				
			}
			cout << doublequote << endl; 
			numObjects++;			
			dup++;
		}
		strs.clear();
	}
	
    }

	ofstream statfile(argv[3]);
	if (statfile.is_open()) {
		statfile << dup << "/" << numObjects << " " << ((double) dup) / numObjects << endl;
		statfile.close();		
	}

    return 0; // success
}

