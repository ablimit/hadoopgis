#include <cmath>
#include "hadoopgis.h"

const float AVG_AREA = 125.0;

vector<string> geometry_collction ; 

void processQuery()
{
    polygon poly;
    polygon hull;
    point center;
    float area,perimeter;

    if (geometry_collction.size()>0)
    {
	for (vector<string>::iterator it = geometry_collction.begin() ; it != geometry_collction.end(); ++it){
	    boost::geometry::read_wkt(*it, poly);
	    area = abs(boost::geometry::area(poly));
	    boost::geometry::centroid(poly, center);
	    boost::geometry::convex_hull(poly, hull);
	    perimeter = boost::geometry::perimeter(poly);

	    cout << area <<comma 
		<< boost::geometry::dsv(center) << comma
		<< boost::geometry::dsv(hull)   << comma 
		<< perimeter << endl;
	}

    }
    cout.flush();
}


int main(int argc, char **argv) {
    string input_line;
    string tile_id ;
    polygon polygon_object ;
    float ar =0.0;


    while(cin && getline(cin, input_line) && !cin.eof()){

        size_t pos = input_line.find_first_of(comma,0);
        if (pos == string::npos)
            return 1; // failure

        tile_id = input_line.substr(0,pos);
        pos=input_line.find_first_of(comma,pos+1);
        string sobject = shapebegin + input_line.substr(pos+2,input_line.length()- pos - 3) + shapeend;
        boost::geometry::read_wkt(sobject, polygon_object);  // spatial filtering
        ar = abs(boost::geometry::area(polygon_object));
        if (ar>AVG_AREA)
            geometry_collction.push_back(sobject);
    }

    processQuery();

    return 0; // success
}

