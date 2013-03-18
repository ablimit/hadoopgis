#include <iostream>
#include <string.h>
#include <string>
#include <boost/unordered_set.hpp>

using namespace std;

const string  UNDERSCORE = "_";
const string  TAB = "\t";
const string POTENTIAL_DUP_STR = "10";
const int DUP_KEY_LENGTH = 2;

int main(int argc, char **argv) {
	boost::unordered_set<string> checkset;
	boost::unordered_set<string>::iterator it;
	std::pair<boost::unordered_set<string>::iterator,bool> ret;

	size_t firstPos;
    string input_line;
    string key;
	string data;
	int irregStart = DUP_KEY_LENGTH + 1;
	bool isOrdinaryReducer = true;

	/* Read the first line to see which type of input it is getting */
	if (cin && getline(cin, input_line) && !cin.eof()) {

		firstPos = input_line.find_first_of(TAB, 0);
		if (input_line.at(DUP_KEY_LENGTH - 1) == 0) {
		// if (POTENTIAL_DUP_STR.compare(input_line.substr(0, firstPos)) == 0) {
			/* Set the reducer to check for duplicates */
			isOrdinaryReducer = false;
			cout << input_line.substr(irregStart);
			/* Insert the element to the checkset */
			checkset.insert(input_line.substr(0, input_line.find_first_of(TAB, 0)));
		} else {
			cout << input_line.substr(DUP_KEY_LENGTH);
		}
	}

	if (isOrdinaryReducer) {
		while(cin && getline(cin, input_line) && !cin.eof()) {
			/* Output regular input */
			cout << input_line.substr(DUP_KEY_LENGTH);
		}	
	} else {
		while(cin && getline(cin, input_line) && !cin.eof()) {
			data = input_line.substr((irregStart));

		    /* Delimits the imageidpolygonids and statistics */
		    firstPos = data.find_first_of(UNDERSCORE, 0);
		    if (string::npos == firstPos)
		        cout << data << endl;
		    else { 
		        firstPos = data.find_first_of(UNDERSCORE, firstPos + 1);
		        if (string::npos == firstPos)
		            cout << data << endl;
		        else {
		            key = data.substr(0, data.find_first_of(TAB, firstPos));
		            ret = checkset.insert(key);
		            if (ret.second == true) { // inserted for the first time
		                /* Output the result */
		                cout << data << endl;			
		            }
		        }
		    }
		}
    }
    return 0;
}

