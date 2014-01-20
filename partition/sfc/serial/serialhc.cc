#include "Timer.hpp"
#include <boost/program_options.hpp>
#include "../../SpaceStreamReader.h"

#include <iostream>
#include <string>
#include <cmath>

//#include <vector>
//#include <algorithm>

using namespace std;
using namespace SpatialIndex;
//using namespace SpatialIndex::RTree;
namespace po = boost::program_options;

int FILLING_CURVE_PRECISION =0; // precision 
class MyVisitor : public IVisitor
{
 public:
  size_t m_indexIO;
  size_t m_leafIO;

 public:
  MyVisitor() : m_indexIO(0), m_leafIO(0) {}

  void visitNode(const INode& n)
  {
    if (n.isLeaf()) m_leafIO++;
    else m_indexIO++;
  }

  void visitData(const IData& d)
  {
    cout << " " << d.getIdentifier() ;
    // the ID of this data entry is an answer to the query. I will just print it to stdout.
  }

  void visitData(std::vector<const IData*>& v)
  {
    cout << v[0]->getIdentifier() << " " << v[1]->getIdentifier() << endl;
  }
};


/**
 * Computes the hilbert value for two given integers.
 * The number of bits of x and y to be considered can be determined by setting the parameter mask appropriately.
 * @param x the value of the first dimension
 * @param y the value of the second dimension
 * @param mask the bitmask containing exactly the highest bit to be considered.
 *  	If all bits of x and y should be taken into account, choose the value 1<<31.
 * @return the hilbert value for x and y
 */
long hilbert2d (int x, int y, unsigned int mask) {
  long hilbert = 0;
  int not_y = ~(y ^= x);

  do
    if ((y&mask)!=0)
      if ((x&mask)==0)
        hilbert = (hilbert<<2)|1;
      else {
        x ^= not_y;
        hilbert = (hilbert<<2)|3;
      }
    else
      if ((x&mask)==0) {
        x ^= y;
        hilbert <<= 2;
      }
      else
        hilbert = (hilbert<<2)|2;
  while ((mask >>= 1)!=0);
  return hilbert;
}

/**
 * Computes the hilbert value for two given integers.
 * @param x the value of the first dimension
 * @param y the value of the second dimension
 * @return the hilbert value for x and y
 */
long hilbert2d (int x, int y) {
  return hilbert2d(x, y, 1<<31);
}

long getHilbertValue(double obj []) {
   double x = (obj[0] + obj[2])/2;
   double y= (obj[1] + obj[3])/2;
  return hilbert2d((int) (x*FILLING_CURVE_PRECISION),(int) (y*FILLING_CURVE_PRECISION));
}

struct HilberCurveSorter {
    bool operator() (double i [] , double j [] ) { 

      return (getHilbertValue(i)< getHilbertValue(j));
    }
} hcsorter;

void printShape(id_type id, double obj []){
  string space = " "; 
  long hcval = getHilbertValue(obj);
  cout << id << space << obj[0] << space << obj[1] << space << obj[2] << space << obj[3] << space << hcval <<endl;
}

int main(int ac, char* av[]){
  cout.precision(15);
  
  uint32_t bucket_size ;
  int prec;
  string inputPath;

  try {
    po::options_description desc("Options");
    desc.add_options()
        ("help", "this help message")
        ("prec,p", po::value<int>(&prec)->default_value(20), "Hilbert Curve Precision")
        ("bucket,b", po::value<uint32_t>(&bucket_size), "Expected bucket size")
        ("input,i", po::value<string>(&inputPath), "Data input file path")
        // ("index", po::value<string>(&indexPath), "index file path")
        ;

    po::variables_map vm;        
    po::store(po::parse_command_line(ac, av, desc), vm);
    po::notify(vm);    

    if ( vm.count("help") || (! vm.count("bucket")) || (!vm.count("input")) || (!vm.count("prec")) ) {
      cerr << desc << endl;
      return 0; 
    }

    cerr << "Precision: "<< prec <<endl;
    cerr << "Bucket size: "<<bucket_size <<endl;
    cerr << "Data: "<<inputPath <<endl;
  }
  catch(exception& e) {
    cerr << "error: " << e.what() << "\n";
    return 1;
  }
  catch(...) {
    cerr << "Exception of unknown type!\n";
    return 1;
  }

  // assign precision 
  FILLING_CURVE_PRECISION = 1<<prec;
  
  // read objects into memory 
  std::ifstream m_fin;
  m_fin.open(inputPath.c_str());
  SpatialIndex::id_type id;
  vector<double*> socoll;  // spatial object collection 
  while (!m_fin.eof())
  {
    double * obj = new double [4]; 
    m_fin >> id >> obj[0] >> obj[1] >> obj[2] >> obj[3] ;
    if (m_fin.good()){ 
      socoll.push_back(obj);
      //printShape(id, obj);
    }
    else 
      delete [] obj; 
  }
  m_fin.close();

   cerr << "About to sort. " << endl;
  Timer t;                         
  // sort object based on Hilbert Curve value
  std::sort (socoll.begin(), socoll.end(), hcsorter); 
  cerr << "sort ------ done. " << endl;
  // partition based ob hilbert curve
  double low[2], high[2];
  vector<RTree::Data*> tiles;
  id_type tid = 1;
  vector<double*>::size_type len = socoll.size();
  for (vector<double*>::size_type i = 0; i < len; i+=bucket_size)
  {
    low[0] = std::numeric_limits<double>::max();
    low[1] = std::numeric_limits<double>::max();
    high[0] = 0.0 ;
    high[1] = 0.0 ;
    for (int j = 0; j < bucket_size-1 && i+j < len; j++)
    {
      //cerr << "i+j = " << i+j << endl;
      double *obj=  socoll[i+j] ;
        if (obj[0] < low[0])  low[0] = obj[0];
        if (obj[1] < low[1])  low[1] = obj[1];
        if (obj[2] > high[0]) high[0] = obj[2];
        if (obj[3] > high[1]) high[1] = obj[3];
    }
      Region r(low, high, 2);
      tiles.push_back(new RTree::Data(0, 0 , r, tid++));
      //cerr << "tid: " << tid << "; " << "i: " << i<< endl;
  }
  double elapsed_time = t.elapsed();
  cerr << "stat:ptime," << bucket_size << "," << tiles.size() <<"," << elapsed_time << endl;

  // build in memory Tree
  SpaceStreamReader stream(inputPath);
  IStorageManager* memoryFile = StorageManager::createNewMemoryStorageManager();
  id_type indexIdentifier;

  t.restart(); // reset timer 

  ISpatialIndex* tree = RTree::createAndBulkLoadNewRTree(
      RTree::BLM_STR, stream, *memoryFile, 0.99, 20, 1000, 2, SpatialIndex::RTree::RV_RSTAR, indexIdentifier);

  elapsed_time = t.elapsed();
  cerr << "stat:itime," << elapsed_time << endl;

  // assign partition id 
  t.restart();
  RTree::Data* d = NULL; 
  for (vector<RTree::Data*>::iterator it = tiles.begin() ; it != tiles.end(); ++it) 
  {
    MyVisitor vis;
    d = *it;
    // this will find all data that intersect with the query range.
    cout << d->getIdentifier();
    tree->intersectsWithQuery(d->m_region, vis);
    cout << endl;
  }
  elapsed_time = t.elapsed();
  cerr << "stat:rtime," << elapsed_time << endl;
  
  // cleanup allocated memory 
  for (vector<double*>::iterator it = socoll.begin() ; it != socoll.end(); ++it) 
    delete [] *it;
  for (vector<RTree::Data*>::iterator it = tiles.begin() ; it != tiles.end(); ++it) 
    delete *it;

  delete tree;
  delete memoryFile;
  return 0;

}

