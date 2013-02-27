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
//#include <boost/geometry/algorithms/centroid.hpp>
//#include <boost/geometry/geometries/adapted/boost_tuple.hpp>

//BOOST_GEOMETRY_REGISTER_BOOST_TUPLE_CS(cs::cartesian)

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

const string shapebegin = "POLYGON((";
const string shapeend = "))";

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
	
		index = input_line[pos+1] - offset;   // the very first character denotes join index
		// pos2 = input_line.find_first_of(tab, pos+1);

		/* Next character is polygon ID */
		tmpid = input_line.substr(pos + 3, 7);

		// pos = input_line.find_first_of(tab, pos +1);
		value = input_line.substr(pos + 11,input_line.size() - pos - 11);
	
		shape = new polygon();
		mbb = new box();
		boost::geometry::read_wkt(shapebegin+value+shapeend,*shape);
		boost::geometry::correct(*shape);
		boost::geometry::envelope(*shape, *mbb);
		//assert(boost::geometry::area(*mbb)>=0);
		outline[key][index].push_back(mbb);
		markup[key][index].push_back(shape);
		polyid[key][index].push_back(tmpid);
    }

    // cerr << "tile size" << "\t markup " << markup.size() << "\t mbb " << outline.size()<< endl;
    return true;
}

int main(int argc, char** argv)
{

    boxmap::iterator iter;
    string key;
	string imageid;
    int size = -1;
	int pos3;
    double overlap_ratio = 0.0 ;
    double overlap_area = 0.0 ;
    double union_area = 0.0 ;
    double distance = 0.0 ;

	int x1, y1, x2, y2;
	x1 = 0;
	y1 = 0;
	x2 = 0;
	y2 = 0;

    point a ;
    point b ;

    if (readSpatialInput())
    {
	// for each key(tile) in the input stream 
	for (iter= outline.begin(); iter != outline.end(); iter++)
	{
	    key  = iter->first;
		pos3 = key.find_first_of(minusS, 0);
		imageid = key.substr(0, pos3);

		// cout << " id first polygon = " << polyid[key][0][0];		
	    map<int ,vector<box*> > &mbbs = outline[key];
	    map<int ,vector<polygon*> > &shapes = markup[key];

	    size = mbbs.size(); // join cardinality 
	    vector<polygon> output;

	    int outerloop_size = (mbbs[0]).size(); //src tile polygon size
	    for (int i =1; i< size ;i++)
	    {
		int innerloop_size = (mbbs[i]).size(); // tar tile polygon size 
		for (int j =0; j< outerloop_size; j++)
		{
		    for (int k=0; k< innerloop_size; k++)
		    {
			if (boost::geometry::intersects( *(mbbs[0][j]), *(mbbs[i][k])) )
			    //&& boost::geometry::intersects(*((*shapes)[0][j]), *((*shapes)[i][k]))) // if the mbb && shapes intersects 
			{
			    boost::geometry::intersection(*(shapes[0][j]),*(shapes[i][k]),output);
			    if (output.empty())
				continue;
			    double area_src = boost::geometry::area(*(shapes[0][j]));
			    double area_tar = boost::geometry::area(*(shapes[i][k]));
			    double overlap  = 0.0;
			    double uni = 0.0;
				
				overlap_area = 0;

			    BOOST_FOREACH(polygon & p, output)
			    {
				boost::geometry::correct(p);
				overlap_area += boost::geometry::area(p);
				//cerr << "intersection: " << overlap << endl; 
				//overlap = boost::geometry::area(p);
				//cerr << "intersection: " << overlap << endl;
			    }
	
			    /* 
			     * cerr<< "poly_area A: " << area_src << endl; 
			     * cerr<< "poly_area B: " << area_tar << endl; 
			     * cerr<< "overlap: " << overlap << endl; 
			     * cerr << "union: " << uni << endl;
			     */

				/* Output statistics */
				cout << imageid << minusS << polyid[key][0][j] << minusS << polyid[key][i][k] << tab << overlap_area << endl;				

				output.clear();
			}
		    }
		}
	    }

	}
    }

    return 0;
}

