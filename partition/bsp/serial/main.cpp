#include <iostream>
#include <sstream>
#include <fstream>
#include <cstring>
#include <cmath>
#include <map>
#include <cstdlib> 
#include "commonspatial.h"
#include "Timer.hpp"

#include <boost/program_options.hpp>
#include <spatialindex/SpatialIndex.h>

using namespace SpatialIndex;
using namespace std;
// using namespace boost;
namespace po = boost::program_options;

// constants 
const string COMMA = ",";
const string SPACE = " ";
const string TAB = "\t";
const string NEW_LINE = "\n";

//function defs
void processInput();
bool readInputFile(string);

// global vars 
int bucket_size ;
vector<RTree::Data*> tiles;
vector<BinarySplitNode*> leafNodeList;
vector<SpatialObject*> listAllObjects;
BinarySplitNode *tree;
stringstream mystr;
int id;
double left, right, top, bottom;

// main method
int main(int ac, char** av) {
  cout.precision(15);
  string inputPath;

  try {
    po::options_description desc("Options");
    desc.add_options()
      ("help", "this help message")
      ("bucket,b", po::value<int>(&bucket_size), "Expected bucket size")
      ("input,i", po::value<string>(&inputPath), "Data input file path");

    po::variables_map vm;        
    po::store(po::parse_command_line(ac, av, desc), vm);
    po::notify(vm);    

    if ( vm.count("help") || (! vm.count("bucket")) || (!vm.count("input")) ) {
      cerr << desc << endl;
      return 0; 
    }

    cerr << "Bucket size: "<<bucket_size <<endl;
    cerr << "Data: "<<inputPath <<endl;
    //return 0; 

  }
  catch(exception& e) {
    cerr << "error: " << e.what() << "\n";
    return 1;
  }
  catch(...) {
    cerr << "Exception of unknown type!\n";
    return 1;
  }
  // argument parsing is done here.


  if (!readInputFile(inputPath)) {
    cerr << "Error reading input in" << endl;
    return -1;
  }

  Timer t; 
  processInput();
  double elapsed_time = t.elapsed();
  cerr << "stat:ptime," << bucket_size << "," << tiles.size()<<"," << elapsed_time << endl;

  //print tile memory and cleanup tiles space 
  for (vector<RTree::Data*>::iterator it = tiles.begin() ; it != tiles.end(); ++it) 
  {
    cout << (*it)->m_id << " " << (*it)->m_region <<endl;
    delete *it;
  }
  tiles.clear();


  // Memeory cleanup here. 
  // delete the stuff inside your vector
  //
  for (vector<SpatialObject*>::iterator it = listAllObjects.begin(); it != listAllObjects.end(); it++) 
    delete *it;

  for(vector<BinarySplitNode*>::iterator it = leafNodeList.begin(); it != leafNodeList.end(); it++ ) {
    delete *it;
  }

  delete tree;

  leafNodeList.clear();
  listAllObjects.clear(); 

  return 0;
}

void processInput() {
  BinarySplitNode *tree = new BinarySplitNode(0.0, 1.0, 1.0, 0.0, 0);
  leafNodeList.push_back(tree);
  for (vector<SpatialObject*>::iterator it = listAllObjects.begin(); it != listAllObjects.end(); it++) { 
    tree->addObject(*it);
  }

  id_type tid=1; //bucket id
  double low[2], high[2];
  //int countLeaf = 0;
  for(vector<BinarySplitNode*>::iterator it = leafNodeList.begin(); it != leafNodeList.end(); it++ ) {
    BinarySplitNode *tmp = *it;
    if (tmp->isLeaf) {
      // cout << ++countLeaf << SPACE << tmp->left << SPACE  << tmp->bottom 
      //    << SPACE << tmp->right << SPACE << tmp->top << SPACE << tmp->size << endl ;
      low[0] = tmp->left ;
      low[1] = tmp->bottom;
      high[0] = tmp->right;
      high[1] = tmp->top;
      Region r(low, high, 2);
      tiles.push_back(new RTree::Data(0, 0 , r, tid++));
    }
  }
}

bool readInputFile(string inputFilePath) {
  ifstream inFile(inputFilePath.c_str());
  string input_line;

  /* Read in the MBBs */

  while (std::getline(inFile, input_line)) {
    mystr.str(input_line);
    mystr.clear();

    SpatialObject *obj = new SpatialObject(0, 0, 0, 0);
    mystr >> id >> obj->left >> obj->bottom >> obj->right >> obj->top;
    listAllObjects.push_back(obj);
  }


  return true;
}


