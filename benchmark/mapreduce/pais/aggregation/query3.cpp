#include <cmath>
#include "hadoopgis.h"

vector<string> geometry_collction ; 

void processQuery()
{
    polygon poly;
    polygon hull;
    point center;
    float avg_area,avg_perimeter;

    if (geometry_collction.size()>0)
    {
	for (vector<string>::iterator it = geometry_collction.begin() ; it != geometry_collction.end(); ++it){
	    boost::geometry::read_wkt(*it, poly);
	    avg_area += abs(boost::geometry::area(poly));
	    //boost::geometry::centroid(poly, center);
	    //boost::geometry::convex_hull(poly, hull);
	    avg_perimeter += boost::geometry::perimeter(poly);

	    /*
	    cout << area <<comma 
		<< boost::geometry::dsv(center) << comma
		<< boost::geometry::dsv(hull)   << comma 
		<< perimeter << endl;
	    */
	}
	    cout << avg_area/geometry_collction.size() << comma << avg_perimeter/geometry_collction.size() << endl;

    }
    cout.flush();
}


int main(int argc, char **argv) {
    string input_line;
    string tile_id ;

        while(cin && getline(cin, input_line) && !cin.eof()){

            size_t pos = input_line.find_first_of(comma,0);
            if (pos == string::npos)
                return 1; // failure

            tile_id = input_line.substr(0,pos);
            pos=input_line.find_first_of(comma,pos+1);
            geometry_collction.push_back(shapebegin + input_line.substr(pos+2,input_line.length()- pos - 3) + shapeend);
            //cout << key<< tab << index<< tab << shapebegin <<value <<shapeend<< endl;
        }

    processQuery();

    return 0; // success
}

