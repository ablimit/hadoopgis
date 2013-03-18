#include <cmath>
#include "hadoopgis.h"

const int tile_size  = 4096;
const string paisUID = "gbm1.1";

vector<string> geometry_collction ; 

bool paisUIDMatch(string pais_uid)
{
    char * filename = getenv("map_input_file");
    //char * filename = "astroII.1.1";
    if ( NULL == filename ){
        cerr << "map.input.file is NULL." << endl;
        return false;
    }

    if (pais_uid.compare(filename) ==0)
        return true;
    else 
        return false;
}
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

	    cout << area <<COMMA 
		<< boost::geometry::dsv(center) << COMMA
		<< boost::geometry::dsv(hull)   << COMMA 
		<< perimeter << endl;
	}

    }
    cout.flush();
}


int main(int argc, char **argv) {
    string input_line;
    string tile_id ;

    if (paisUIDMatch(paisUID))
    {
        while(cin && getline(cin, input_line) && !cin.eof()){

            size_t pos = input_line.find_first_of(COMMA,0);
            if (pos == string::npos)
                return 1; // failure

            tile_id = input_line.substr(0,pos);
            pos=input_line.find_first_of(COMMA,pos+1);
            geometry_collction.push_back(shapebegin + input_line.substr(pos+2,input_line.length()- pos - 3) + shapeend);
            //cout << key<< tab << index<< tab << shapebegin <<value <<shapeend<< endl;
        }
    }

    processQuery();

    return 0; // success
}

