#include <iostream>
#include <string.h>
#include <string>
#include <vector>

#include <boost/algorithm/string.hpp>

using namespace std;
using namespace std;
using boost::lexical_cast;

int GEOM_IDX = 4;
int ID_IDX = 3;
int TILE_IDX = 2;
int SET_IDX = 1;
int IMAGE_IDX = 0;


vector<string> parse(string & line) {
    vector<string> vec ;
    boost::split(vec, line, boost::is_any_of(BAR));
    return vec;
}

int main(int argc, char **argv) {

    string DEL = "|";
    string TAB = "\t";
    string DASH = "-";
    size_t pos ;
    string image_name;
    string input_line;
    vector<string> fields;

    while(cin && getline(cin, input_line) && !cin.eof()){
	fields = parse(input_line);
	pos = fields[IMAGE_IDX].find_first_of(DASH);
	image_name = fields[IMAGE_IDX].substr(0,pos);
	cout << image_name << fields[TILE_IDX] << TAB << fields[SET_IDX] << BAR << fields[ID_IDX] <<BAR << fields[GEOM_IDX] <<endl;
	fields.clear();
    }
    cout.flush();

    return 0; // success
}

