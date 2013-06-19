#include <iostream>
#include <string.h>
#include <string>
#include <vector>

#include <boost/algorithm/string.hpp>

using namespace std;

int GEOM_IDX = 2;
int ID_IDX = 1;
int TILE_IDX = 0;

string DEL = "|";
string TAB = "\t";
string DASH = "-";

int getJoinIndex ()
{
    char * filename = getenv("map_input_file");
    //char * filename = "astroII.1.1";
    if ( NULL == filename ){
	cerr << "map.input.file is NULL." << endl;
	return 1 ;
    }
    int len= strlen(filename);
    int index = filename[len-1] - '0' ;
    return index;
}

vector<string> parse(string & line) {
    vector<string> vec ;
    boost::split(vec, line, boost::is_any_of(DEL));
    return vec;
}

int main(int argc, char **argv) {
    int index = getJoinIndex();
    // cerr << "Index: " << index << endl; 
    if (index <1)
    {
	cerr << "InputFileName index is corrupted.. " << endl;
	return 1; //failure 
    }

    size_t pos ;
    string image_name;
    string input_line;
    vector<string> fields;

    while(cin && getline(cin, input_line) && !cin.eof()){
	fields = parse(input_line);
	cout << fields[TILE_IDX] << TAB << index << DEL << fields[ID_IDX] <<DEL << fields[GEOM_IDX] <<endl;
	fields.clear();
    }
    cout.flush();


    return 0; // success
}

