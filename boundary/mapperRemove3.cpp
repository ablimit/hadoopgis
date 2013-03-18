#include <iostream>
#include <string.h>
#include <string>
#include <cstdlib>
#include <cmath>

using namespace std;

const int POTENTIAL_DUP = 100;
const string POTENTIAL_DUP_STR = "100";
const string TAB = "\t";
const string  UNDERSCORE = "_";
const string DASH = "-";
const int MOD = 99;

int main(int argc, char **argv) {
	size_t firstPos;
	size_t secondPos;
    string input_line;
    string key;
	int keyDup;
	// int regularKey = (rand() % POTENTIAL_DUP);

    while(cin && getline(cin, input_line) && !cin.eof()) {
        /* Delimits the imageidpolygonids and statistics */
        firstPos = input_line.find_first_of(UNDERSCORE, 0);
        if (string::npos == firstPos) {
			/* Assign a random key between 0 and POTENTIAL_DUP - 1 inclusive */
            cout << (rand() % POTENTIAL_DUP) << TAB << input_line << endl;
			// cout << (rand() % POTENTIAL_DUP) << TAB << input_line << endl;
        } else { 
            firstPos = input_line.find_first_of(UNDERSCORE, firstPos+1);
            if (string::npos == firstPos) {
                 cout << rand() % POTENTIAL_DUP << TAB << input_line << endl;
            } else {
				 secondPos = input_line.find_last_of(DASH, firstPos);
				 keyDup = boost::lexical_cast<int>(input_line.substr(secondPos + 1, firstPos - secondPos - 1)) % MOD + POTENTIAL_DUP;
                 cout << keyDup << TAB << input_line << endl;	
            }
        }
    }
}
