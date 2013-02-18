#include "hadoopgis.h"

using namespace std;

const string paisUID = "gbm1.1";
const string tileID = "gbm1.1-0000040960-0000040960";

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
	return false
}

int main(int argc, char **argv) {
    string input_line;
    string tile_id ;
    string polygon;
    string 

    if (paisUIDMatch(paisUID))
    {
	while(cin && getline(cin, input_line) && !cin.eof()){

	    size_t pos=input_line.find_first_of(comma,0);
	    if (pos == string::npos)
		return 1; // failure

	    tile_id = input_line.substr(0,pos);
	    if (tileID.compare(tile_id)==0) // if tile ID matches, continue searching 
	    {
		pos=input_line.find_first_of(comma,pos+1);
		polygon = shapebegin + input_line.substr(pos+2,input_line.length()-pos-3) + shapeend;

		//cout << key<< tab << index<< tab << shapebegin <<value <<shapeend<< endl;
	    }
	}
    }

    cout.flush();

    return 0; // success
}

