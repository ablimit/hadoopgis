#include <cmath>
#include "hadoopgis.h"

vector<string> geometry_collction ; 

void processQuery()
{
    polygon poly;
    polygon hull;
    point center;
    double avg_area = 0.0 ;
    double avg_perimeter = 0.0;
    int cc =0; 
    if (geometry_collction.size()>0)
    {
	for (vector<string>::iterator it = geometry_collction.begin() ; it != geometry_collction.end(); ++it){
	    boost::geometry::read_wkt(*it, poly);
	    avg_area = abs(boost::geometry::area(poly));
	    //boost::geometry::centroid(poly, center);
	    //boost::geometry::convex_hull(poly, hull);
	    avg_perimeter = boost::geometry::perimeter(poly);

	    /*
	    cout << area <<COMMA 
		<< boost::geometry::dsv(center) << COMMA
		<< boost::geometry::dsv(hull)   << COMMA 
		<< perimeter << endl;
	    */
	    cout << cc%100 <<TAB << avg_area << TAB << avg_perimeter << endl;
	}
    }
    cout.flush();
}


int main(int argc, char **argv) {
    string input_line;
    string tile_id ;

        while(cin && getline(cin, input_line) && !cin.eof()){

            size_t pos = input_line.find_first_of(COMMA,0);
            if (pos == string::npos)
                return 1; // failure

            tile_id = input_line.substr(0,pos);
            pos=input_line.find_first_of(COMMA,pos+1);
            geometry_collction.push_back(shapebegin + input_line.substr(pos+2,input_line.length()- pos - 3) + shapeend);
            //cout << key<< TAB << index<< TAB << shapebegin <<value <<shapeend<< endl;
        }

    processQuery();

    return 0; // success
}

