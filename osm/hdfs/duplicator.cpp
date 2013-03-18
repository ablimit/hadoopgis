//#include <stdlib.h>     /* atoi */
#include <iostream>
#include <string>
#include <vector>

#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/join.hpp>

#define OSM_ID 0
#define OSM_TILEID 1
#define OSM_OID 2
#define OSM_ZORDER 3
#define OSM_POLYGON 4

using namespace std;


const string BAR = "|";
const string UNDERSCORE = "_";

int getJoinIndex ()
{
    char * filename = getenv("map_input_file");
    //char * filename = "astroII.1.1";
    if ( NULL == filename ){
	cerr << "map.input.file is NULL." << endl;
	return 1 ;
    }
    int len= strlen(filename);
    int index = filename[len-1] - '1' ;
    return index;
}


int main(int argc, char **argv) {

    if (argc< 2)
    {
	cerr << "Usage: " << argv[1] << " [Duplication Factor]" <<endl;
	return -1;
    }
    const int dupFactor  = atoi(argv[1]);
    const int index = getJoinIndex();

    string input_line;
    vector<string> fields;

    while(cin && getline(cin, input_line) && !cin.eof()){	
	boost::split(fields, input_line, boost::is_any_of(BAR));
	if (dupFactor >1)
	{
	    for (int i=0; i<dupFactor;i++ )
	    {
		cout << index<< UNDERSCORE << i << UNDERSCORE<< boost::algorithm::join(fields,BAR) << endl;
	    }
	}
	else 
	{
	    cout << input_line<<endl;
	}
	
	fields.clear();
    }

    cout.flush();

    return 0; // success
}

