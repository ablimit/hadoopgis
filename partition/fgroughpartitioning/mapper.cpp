#include <iostream>
#include <sstream>
#include <fstream>
#include <cstring>
#include <cmath>
#include <map>
#include <cstdlib> 
#include <getopt.h>

using namespace std;

stringstream mystr;

bool extractParams(int argc, char** argv);
bool readInput();

const int DEFAULT_KEY = 0;
const string TAB = "\t";
const string SPACE = " ";

double min_x;
double max_x;
double min_y;
double max_y;
double x_length;
double y_length;
int x_split;
int y_split;
long **arr;

int main(int argc, char** argv) {
	/* Parsing argument here */
	int i, j;
	if (!extractParams(argc, argv)) {
		return -1;
	}


	arr = new long*[x_split];
	for (i = 0; i < x_split; i++) {
		arr[i] = new long[y_split];
	}

	/*
	   for (i = 0; i < x_split; i++) {
	   for (j = 0; j < y_split; j++) {
	   cerr << arr[i][j] << endl;
	   }
	   }
	 */

	readInput();

	/* Output to reducer */
	cout << DEFAULT_KEY << TAB;
	for (i = 0; i < x_split; i++) {
		for (j = 0; j < y_split; j++) {
			cout << arr[i][j] << SPACE;
		}
	}

	cout << endl;

	for (i = 0; i < x_split; i++) {
		delete[] arr[i];
	}

	delete[] arr;
	return 0;
}

bool extractParams(int argc, char** argv ) { 
	/* getopt_long stores the option index here. */
	int option_index = 0;
	/* getopt_long uses opterr to report error*/
	opterr = 0 ;
	struct option long_options[] =
	{
		{"minX", required_argument, 0, 'l'},
		{"maxX", required_argument, 0, 'r'},
		{"minY", required_argument, 0, 'b'},
		{"maxY", required_argument, 0, 't'},
		{"numXsplits", required_argument, 0, 'x'},
		{"numYsplits", required_argument, 0, 'y'},
		{0, 0, 0, 0}
	};

	int c;

	while ((c = getopt_long (argc, argv, "l:r:b:t:x:y:",long_options, &option_index)) != -1){
		switch (c)
		{
			case 0:
				/* If this option set a flag, do nothing else now. */
				if (long_options[option_index].flag != 0)
					break;
				cout << "option " << long_options[option_index].name ;
				if (optarg)
					cout << "a with arg " << optarg ;
				cout << endl;
				break;

			case 'l':
				min_x = atof(optarg);
				break;
			case 'r':
				max_x = atof(optarg);
				break;
			case 'b':
				min_y = atof(optarg);
				break;
			case 't':
				max_y = atof(optarg);
				break;

			case 'x':
				x_split = strtol(optarg, NULL, 10);
				break;
			case 'y':
				y_split = strtol(optarg, NULL, 10);
				break;

			case '?':
				return false;
				/* getopt_long already printed an error message. */
				break;

			default:
				return false;
		}

	}

	x_length = (max_x - min_x ) / x_split;
	y_length = (max_y - min_y ) / y_split;

	return true;
}

bool readInput() {
	string input_line;
	long id;
	double left, right, top, bottom;
	int tile_x_min, tile_x_max, tile_y_min, tile_y_max;
	int i, j;
	/* Read in the MBBs */
	int count = 0;
	int countboundary = 0;
	int total = 0;

	while (std::getline(cin, input_line)) {
		mystr.str(input_line);
		mystr.clear();

		mystr >> id >> left >> bottom >> right >> top;

		tile_x_min = (int) ((left - min_x) / x_length);
		tile_x_max = (int) ((right - min_x) / x_length);
		tile_y_min = (int) ((bottom - min_y) / y_length);
		tile_y_max = (int) ((top - min_y) / y_length);

		/*
		   if (tile_x_min < tile_x_max || tile_y_min < tile_y_max) {
		//cerr << "boundary object: " << countboundary++ << SPACE << tile_x_min << SPACE << tile_x_max << SPACE << tile_y_min << SPACE << tile_y_max << endl;
		countboundary++;
		} else {
		// cerr << "not boundary" << endl;
		} */

		/* Perform the counting */
		for (i = tile_x_min; i <= tile_x_max; i++) {
			for (j = tile_y_min; j <= tile_y_max; j++) {
				arr[i][j]++;
				//total++;
				//cerr << count++ << SPACE << i << SPACE << j << endl;
			} 
		}
	}
	//cerr << "Boundary: " << countboundary << endl;
	//cerr << "Total: " << total << endl;

	return true;
}

