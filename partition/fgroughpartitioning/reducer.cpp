#include <iostream>
#include <sstream>
#include <fstream>
#include <cstring>
#include <cmath>
#include <map>
#include <list>
#include <cstdlib> 
#include <getopt.h>

using namespace std;

stringstream mystr;

bool extractParams(int argc, char** argv);
bool readInput();
void performSplit();
long getCount(int minX, int minY, int width, int height);
int getSplitPoint(struct stnode *tmpNode);
void updateCount(struct stnode *tmpNode);
void outputPolygonWkt(int id, double x_min, double y_min, 
	double x_max, double y_max);

/*  ADJUST FOR SPECIFIC DATASET!
 * Keep this as true if operating on OSM data set:
 * Will convert the normalized [0, 1][0, 1] to [-180,180][-90,90] 
 */
const bool convertOsm = true;
const double MIN_X = -180;
const double MAX_X = 180;
const double MIN_Y = -90;
const double MAX_Y = 90;

const int DEFAULT_KEY = 0;
const string TAB = "\t";
const string SPACE = " ";
const string PREFIX_POLYGON = "POLYGON((";
const string SUFFIX_POLYGON = "))";
const string COMMA = ",";

double min_x;
double max_x;
double min_y;
double max_y;
double x_length;
double y_length;
int x_split;
int y_split;
int num_split;
long **arr;
long total;

struct stnode {
	int min_x;
	int min_y;
	int width;
	int height;
	long count;
	bool splitable;
} node;

list<struct stnode*> listNodes;


int main(int argc, char** argv) {
	/* Parsing argument here */
	int i, j;
	struct stnode *tmpNode;
	struct stnode *newNode;
	struct stnode *maxNode;
	long maxCount;
	int minIdx;
	int count = 0;

	
	if (!extractParams(argc, argv)) {
		// cerr  << "Error extracting param" << endl;
		return -1;
	}


	arr = new long*[x_split];
	for (i = 0; i < x_split; i++) {
		arr[i] = new long[y_split];
	}

	/*
	   for (i = 0; i < x_split; i++) {
	   for (j = 0; j < y_split; j++) {
	//  // cerr  << arr[i][j] << endl;
	}
	} */

	readInput();

		
	/* Output to reducer */
	//cout << DEFAULT_KEY << TAB;
	/*	
	for (i = 0; i < x_split; i++) {
		for (j = 0; j < y_split; j++) {
			cerr  << "i = " << i << " j = " << j << " and count = ";
			cerr  << arr[i][j] << SPACE;
			cerr  << endl;
		}
	}
	*/

	struct stnode *first = new struct stnode;
	first->min_x = 0;
	first->width = x_split;
	first->min_y = 0;
	first->height = y_split;
	first->count = total;
	// cerr  << "total: " << total << endl;
	first->splitable = true;
 
	listNodes.push_back(first);

	int splitPoint;
	for (i = 1; i < num_split; i++) {
		while (true) {
			maxCount = 0;

			for(list<struct stnode*>::iterator it = listNodes.begin(); it != listNodes.end(); it++ ) {
				tmpNode = *it;
				if (!tmpNode->splitable) {
					continue;
				}

				if (j == 0 || ( (tmpNode->width >= 2 || tmpNode->height >= 2) && maxCount < tmpNode->count)) {
					maxCount = tmpNode->count;
					maxNode = tmpNode;
					//cerr << maxCount << SPACE << tmpNode->count << endl;
				}
			}
			/* Node to be split */
			// cerr  << "before split" << endl;			
			

			splitPoint = getSplitPoint(maxNode);
			// cerr  << "splitPoint = " << splitPoint << endl;
			/* create new region */
			newNode = new struct stnode;
			newNode->splitable = true;
			if (splitPoint > 0) {
				/* Split along the y-axis */
				newNode->min_x = maxNode->min_x + splitPoint;
				newNode->min_y = maxNode->min_y;
				newNode->width = maxNode->width - splitPoint;
				newNode->height = maxNode->height;
				updateCount(newNode);
				/* Adjust the old node */
				maxNode->width = splitPoint;
				maxNode->count -= newNode->count;
				// cerr  << "New counts" << maxNode->count << SPACE << newNode->count << endl;

			} else if (splitPoint < 0) {
				/* Split along the x-axis */
				newNode->min_x = maxNode->min_x;
				newNode->min_y = maxNode->min_y - splitPoint;
				newNode->width = maxNode->width;
				newNode->height = maxNode->height + splitPoint;
				updateCount(newNode);
			
				maxNode->height = -splitPoint;
				maxNode->count -= newNode->count;
				// cerr  << "New counts" << maxNode->count << SPACE << newNode->count << endl;
			}
			if (splitPoint) {
				listNodes.push_back(newNode);
				break;
			} else {
				delete newNode;
			}

		}
	}

	/* Output the result */
	for(list<struct stnode*>::iterator it = listNodes.begin(); it != listNodes.end(); it++ ) {
		tmpNode = *it;
		
		outputPolygonWkt(count++, min_x + (tmpNode->min_x) * x_length,
						min_y + (tmpNode->min_y) * y_length,
						min_x + (tmpNode->min_x + tmpNode->width ) * x_length,
						min_y + (tmpNode->min_y + tmpNode->height ) * y_length);
		/* cout << ++count 
			<< SPACE << min_x + (tmpNode->min_x) * x_length
			<< SPACE << min_y + (tmpNode->min_y) * y_length
			<< SPACE << min_x + (tmpNode->min_x + tmpNode->width ) * x_length
			<< SPACE << min_y + (tmpNode->min_y + tmpNode->height ) * y_length 
			<< SPACE << tmpNode->count 
			<< endl;
		*/
	}


	for(list<struct stnode*>::iterator it = listNodes.begin(); it != listNodes.end(); it++ ) {
		delete *it;
	}
	listNodes.clear();

	for (i = 0; i < x_split; i++) {
		delete[] arr[i];
	}

	delete[] arr;
	return 0;
}

/* Output the wkt format for MBB of the region */
void outputPolygonWkt(int id, double x_min, double y_min, 
	double x_max, double y_max) {

	if (convertOsm) {
		x_min = x_min * (MAX_X - MIN_X) + MIN_X;
		x_max = x_max * (MAX_X - MIN_X) + MIN_X;
		y_min = y_min * (MAX_Y - MIN_Y) + MIN_Y;
		y_max = y_max * (MAX_Y - MIN_Y) + MIN_Y;
	}


	cout << id << TAB << PREFIX_POLYGON
		<< x_min << SPACE << y_min << COMMA
		<< x_min << SPACE << y_max << COMMA
		<< x_max << SPACE << y_max << COMMA
		<< x_min << SPACE << y_max << COMMA
		<< x_min << SPACE << y_min
		<< SUFFIX_POLYGON << endl;
}


/* 0 means failure
   positive number means split along the y-axis (x-coordinate of split line)
   negative number means split along the x-axis (y-coordinate of split line)
*/
int getSplitPoint(struct stnode *tmpNode) {
	/* Check for the split across the y-axis */
	long totalCount = tmpNode->count;
	double currentBestRatio = -1;
	double currentRatio = -1;

	int bestsplit = 0;
	int i, j;

	// cerr  << "Splitting node " << tmpNode->min_x << SPACE << tmpNode->min_y << SPACE << tmpNode->width << SPACE << tmpNode->height << SPACE <<tmpNode->count << endl;

	long currentCount = 0;
	for (i = 0; i < tmpNode->width; i++) {
		for (j = 0; j < tmpNode->height; j++) {
			// // cerr  << "idx: "<< tmpNode->min_x + i << SPACE << tmpNode->min_y + j << endl;
			currentCount += arr[tmpNode->min_x + i][tmpNode->min_y + j];
		}
		// cerr  << "i = " << i << " and currentCount = " << currentCount << endl;

		if (currentCount != 0 && currentCount != totalCount) {
			/* No zero count on either side */
			currentRatio = currentCount < totalCount - currentCount ?
				(double) currentCount / (double) (totalCount - currentCount) :
				(double) (totalCount - currentCount) / (double) currentCount;
			if (currentRatio > currentBestRatio) {
				/* Closer to 1 -> better split */
				currentBestRatio = currentRatio;
				bestsplit = i + 1;
			}
		}
	}
	
	/* Check for the split across the x-axis */
	currentCount = 0;
	for (j = 0; j < tmpNode->height; j++) {
		for (i = 0; i < tmpNode->width; i++) {
			currentCount += arr[tmpNode->min_x + i][tmpNode->min_y + j];
		}
		// cerr  << "j = " << j << " and currentCount = " << currentCount << endl;

		if (currentCount != 0 && currentCount != totalCount) {
			/* No zero count on either side */
			currentRatio = currentCount < totalCount - currentCount ?
				(double) currentCount / (double) (totalCount - currentCount) :
				(double) (totalCount - currentCount) / (double) currentCount;
			if (currentRatio > currentBestRatio) {
				/* Closer to 1 -> better split */
				currentBestRatio = currentRatio;
				bestsplit = - (j + 1);
			}
		}
	}
	if (bestsplit == 0) {
		tmpNode->splitable = false;
	}
	// cerr  << "best split found: " << bestsplit << endl;

	return bestsplit;
}

void updateCount(struct stnode *tmpNode) {
	int i, j;
	long tmpCount = 0;
	for (i = 0; i < tmpNode->width; i++) {
		for (j = 0; j < tmpNode->height; j++) {
			tmpCount += arr[tmpNode->min_x + i][tmpNode->min_y + j];
		}
	}
	// cerr  << "updated count: "<< tmpNode->min_x << SPACE << tmpNode->min_y << SPACE << tmpNode->width << SPACE << tmpNode->height << SPACE << tmpCount << endl;
	tmpNode->count = tmpCount;
}

long getCount(int minX, int minY, int width, int height) {
	int i, j;
	long tmpCount = 0;
	for (i = 0; i < width; i++) {
		for (j = 0; j < height; j++) {
			tmpCount += arr[minX + i][minY + j];
		}
	}
	return tmpCount;
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
		{"numSplits", required_argument, 0, 'n'},
		{0, 0, 0, 0}
	};

	int c;

	while ((c = getopt_long (argc, argv, "l:r:b:t:x:y:n:",long_options, &option_index)) != -1){
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

			case 'n':
				num_split = strtol(optarg, NULL, 10);
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

	int i, j;
	long tmpNum;	
	total = 0;

	while (std::getline(cin, input_line)) {
		mystr.str(input_line.substr(2));
		mystr.clear();

		for (i = 0; i < x_split; i++) {
			for (j = 0; j < y_split; j++) {
				mystr >> tmpNum;
				arr[i][j] += tmpNum;
				total += tmpNum;
			}
		}
	}

	return true;
}

