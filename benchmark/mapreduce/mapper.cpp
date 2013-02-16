#include <iostream>
#include <string.h>
#include <string>

using namespace std;

int getJoinIndex (char * filename)
{
    //char * filename = "astroII.1.1";
    if ( NULL == filename ){
	cerr << "map.input.file is NULL." << endl;
	return 1 ;
    }
    int index = filename[0] - '0' ;
    return index;
}

int main(int argc, char **argv) {

    if (argc < 2)
    {
	cerr << "Missing the dataset id as param." << endl;
	return 0;
    }

    int index = getJoinIndex(argv[1]);
    cerr << "Dataset id: " << index << endl; 

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
	value= input_line.substr(pos+1);
	// cout << index << key<< tab << value << endl;
	cout << key<< tab << index<< tab << value << endl;
    }

    return 0; // success
}

