#include <iostream>
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
typedef map<string,map<int,vector<string> > > polynamesmap;

polymap markup;
boxmap outline;
polynamesmap polyid;

const char offset = '1';
const string tab = "\t";
const char comma = ',';
const char underscore = '_';
const char minusS = '-';

bool readSpatialInput();

bool readSpatialInput() {
    string input_line;

    polygon * shape= NULL;
    box * mbb = NULL;
	size_t pos;
	size_t pos2;    
	int index =0;

    string key ;
    string value;
	string tmpid;

    while(cin && getline(cin, input_line) && !cin.eof()) {
		pos = input_line.find_first_of(tab);
		key = input_line.substr(0,pos);
	
		/* The very first character denotes join index */
		index = input_line[pos+1] - offset;   

		/* Next following is polygon ID */
		tmpid = input_line.substr(pos + 3, 6);

		/* Next is the markup in POLYGON((X,Y...)) format */
		value = input_line.substr(pos + 10,input_line.size() - pos - 10);
	
		shape = new polygon();
		mbb = new box();

		/* Read/convert into polygon and mbb object */
		boost::geometry::read_wkt(value,*shape);
		boost::geometry::correct(*shape);
		boost::geometry::envelope(*shape, *mbb);
		
		/* Push everything in a map so we can retrieve everything later */
		outline[key][index].push_back(mbb);
		markup[key][index].push_back(shape);
		polyid[key][index].push_back(tmpid);
    }
    return true;
}

int main(int argc, char** argv)
{
    boxmap::iterator iter;
    string key;
	string imageid;
    int size = -1;
	int pos3;

	double overlap_area;

	int x1, y1, x2, y2;
	x1 = 0;
	y1 = 0;
	x2 = 0;
	y2 = 0;

    point a, b;

    if (readSpatialInput())
    {
		/* for each key (tile-id) in the input stream */
		for (iter= outline.begin(); iter != outline.end(); iter++)
		{
			key  = iter->first;
			pos3 = key.find_first_of(minusS, 0);
			imageid = key.substr(0, pos3);
	
			map<int ,vector<box*> > &mbbs = outline[key];
			map<int ,vector<polygon*> > &shapes = markup[key];

			size = mbbs.size(); /* join cardinality */
			vector<polygon> output;

			int outerloop_size = (mbbs[0]).size(); /* source tile polygon size */
			for (int i =1; i< size ;i++)
			{
			int innerloop_size = (mbbs[i]).size(); /* target tile polygon size */
			for (int j =0; j< outerloop_size; j++)
			{
				for (int k=0; k< innerloop_size; k++)
				{
				if (boost::geometry::intersects( *(mbbs[0][j]), *(mbbs[i][k])) )
				{
					/* The two minimum bounding rectangles intersect */

					boost::geometry::intersection(*(shapes[0][j]),*(shapes[i][k]),output);
					if (output.empty())
					{
						/* Skip if the two polygons do not intersect */
						continue;
					}
					/* Read the overlap area */
					BOOST_FOREACH(polygon & p, output)
					{
					boost::geometry::correct(p);
					overlap_area += boost::geometry::area(p);
					}

					/* Output statistics: overlap area */
					cout << imageid << minusS << polyid[key][0][j] 
						<< minusS << polyid[key][i][k] 
						<< tab << overlap_area << endl;	
					/* Reset the intersecting shapes  */		
					output.clear();
				}
				}
			}
			}
		}
    }
    return 0;
}

