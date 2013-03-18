#include <iostream>
#include <string.h>
#include <string>
#include <cstdlib>

using namespace std;

const int POTENTIAL_DUP = 10;
const string POTENTIAL_DUP_STR = "10";
const string TAB = "\t";
const string  UNDERSCORE = "_";

int main(int argc, char **argv) {
	size_t firstPos;
    string input_line;
    string key;

	key = rand() % POTENTIAL_DUP;

    while(cin && getline(cin, input_line) && !cin.eof()) {
        /* Delimits the imageidpolygonids and statistics */
        firstPos = input_line.find_first_of(UNDERSCORE, 0);
        if (string::npos == firstPos) {
			/* Assign a random key between 0 and POTENTIAL_DUP - 1 inclusive */
            cout << key << TAB << input_line << endl;
        } else { 
            firstPos = input_line.find_first_of(UNDERSCORE, firstPos+1);
            if (string::npos == firstPos) {
                 cout << rand() % POTENTIAL_DUP << TAB << input_line << endl;
            } else {
                 cout << POTENTIAL_DUP_STR << TAB << input_line << endl;	
            }
        }
    }
}
