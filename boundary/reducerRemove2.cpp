#include <iostream>
#include <string.h>
#include <string>
#include <boost/unordered_set.hpp>

using namespace std;

const string  UNDERSCORE = "_";
const string  TAB = "\t";

int main(int argc, char **argv) {
	boost::unordered_set<string> checkset;
	boost::unordered_set<string>::iterator it;
	std::pair<boost::unordered_set<string>::iterator,bool> ret;

	size_t firstPos;
    string input_line;
    string key;

    while(cin && getline(cin, input_line) && !cin.eof()) {
        /* Delimits the imageidpolygonids and statistics */
        firstPos = input_line.find_first_of(UNDERSCORE, 0);
        if (string::npos == firstPos)
            cout << input_line << endl;
        else { 
            firstPos = input_line.find_first_of(UNDERSCORE, firstPos+1);
            if (string::npos == firstPos)
                cout << input_line << endl;
            else {
                key = input_line.substr(0, input_line.find_first_of(TAB, firstPos));
                ret = checkset.insert(key);
                if (ret.second == true) { // inserted for the first time
                    /* Output the result */
                    cout << input_line << endl;			
                }
            }
        }
    }
    return 0;
}

