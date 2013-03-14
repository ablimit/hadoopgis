#include <iostream>
#include <string.h>
#include <string>

using namespace std;

int main(int argc, char **argv) {
    string input_line;
    
    while(cin && getline(cin, input_line) && !cin.eof()) {
		cout << input_line << endl;
    }
    return 0; // success
}
