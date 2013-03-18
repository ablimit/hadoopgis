#include <iostream>
#include <string.h>
#include <string>
#include <boost/unordered_set.hpp>

using namespace std;

const string  UNDERSCORE = "_";
const string  TAB = "\t";
const string POTENTIAL_DUP_STR = "100";
const int DUP_KEY_LENGTH = 3;
const int CHECK_POST = 2;

int main(int argc, char **argv) {
	boost::unordered_set<string> checkset;
	boost::unordered_set<string>::iterator it;
	std::pair<boost::unordered_set<string>::iterator,bool> ret;

	size_t firstPos;
    string input_line;the
    string key;
	string data;

	while(cin && getline(cin, input_line) && !cin.eof()) {
			data = input_line.substr((DUP_KEY_LENGTH + 1));

		    /* Delimits the imageidpolygonids and statistics */
		    firstPos = data.find_first_of(UNDERSCORE, 0);
		    if (string::npos == firstPos) {
		        cout << data << endl;
			} else { 
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
    return 0;
}

