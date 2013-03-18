#include <iostream>
#include <string.h>
#include <string>
#include <cstring>

using namespace std;

int main(int argc, char **argv) {
    string input_line;
	string lastSeen = "";

	/* Read the first line to see which type of input it is getting */
	while (cin && getline(cin, input_line) && !cin.eof()) {
		if (lastSeen.compare(input_line) != 0) {
			cout << input_line << endl;
			lastSeen = input_line;
		}
	}
 
    return 0;
}

