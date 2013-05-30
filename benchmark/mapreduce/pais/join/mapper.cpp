#include <iostream>
#include <string.h>
#include <string>

using namespace std;

const string shapebegin = "POLYGON((";
const string shapeend = "))";

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

int main(int argc, char **argv) {
    int rep = atoi(argv[1]);

    int index = getJoinIndex();
    // cerr << "Index: " << index << endl; 
    if (index <1)
    {
	cerr << "InputFileName index is corrupted.. " << endl;
	return 1; //failure 
    }

    string tab = "\t";
    char comma = ',';
    string input_line;
    string key ;
    string value;

    while(cin && getline(cin, input_line) && !cin.eof()){

	size_t pos=input_line.find_first_of(comma,0);
	if (pos == string::npos)
	    return 1; // failure

	key = input_line.substr(0,pos);
	pos=input_line.find_first_of(comma,pos+1);
	value= input_line.substr(pos+2,input_line.length()-pos-3);
	// cout << index << key<< tab << value << endl;
	for (int i=0; i < rep ; i++)
	cout << key<< "_" <<i<< tab << index<< tab << shapebegin <<value <<shapeend<< endl;
    }
    cout.flush();

    return 0; // success
}

