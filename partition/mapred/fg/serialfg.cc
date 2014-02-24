#include<cmath>
#include <boost/program_options.hpp>
#include "StdinStreamReader.h"
#include "Timer.hpp"

vector<RTree::Data*> tiles;

// using namespace boost;
namespace po = boost::program_options;
using namespace SpatialIndex::RTree;

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

void genTiles(Region &universe, const uint32_t bucket_size, const uint64_t ds) {
  double P = std::ceil(static_cast<double>(ds) / static_cast<double>(bucket_size));
  double split  =std::ceil(std::sqrt(P));
  uint64_t LOOP =static_cast<uint64_t>(std::ceil(std::sqrt(P)));

  cerr << "ds: " << ds << ", partition size:  "<< bucket_size << endl;
  cerr << "split: " << split << ", LOOP "<< LOOP <<endl;
  double width =  (universe.getHigh(0)- universe.getLow(0)) / split;
  //cerr << "Tile width" << SPACE <<width <<endl;
  double height = (universe.getHigh(1)- universe.getLow(1)) / split;
  //cerr << "Tile height" << SPACE << height <<endl;

  double min_x = universe.getLow(0);
  double min_y = universe.getLow(1);
  string SPACE = " ";
  id_type tid = 1;

  double low[2], high[2];

  for (uint32_t i =0 ; i < LOOP ; i++)
  {
    for (uint32_t j =0 ; j< LOOP; j++)
    {
      low[0]  = min_x + i * width ;
      low[1]  = min_y+ j * height ; 
      high[0] = min_x + (i+1) * width;
      high[1] = min_y + (j+1) * height;
      Region r(low, high, 2);
      tiles.push_back(new RTree::Data(0, 0 , r, tid++));
    }
  }
}

int main(int ac, char* av[]){
  uint32_t bucket_size ;
  int indexCapacity ;
  int leafCapacity ;
  double fillFactor ;
  //string inputPath;
  string indexPath;

  try {
    po::options_description desc("Options");
    desc.add_options()
        ("help", "this help message")
        ("indexCap", po::value<int>(&indexCapacity)->default_value(20), "RTree index page size")
        ("leafCap", po::value<int>(&leafCapacity)->default_value(1000), "RTree leaf page size")
        ("fillCap", po::value<double>(&fillFactor)->default_value(0.99), "Page fill factor.")
        ("bucket,b", po::value<uint32_t>(&bucket_size), "Expected bucket size");
        //("input,i", po::value<string>(&inputPath), "Data input file path")
        // ("index", po::value<string>(&indexPath), "index file path")
        ;

    po::variables_map vm;        
    po::store(po::parse_command_line(ac, av, desc), vm);
    po::notify(vm);    

    if ( vm.count("help") || (! vm.count("bucket"))) {
      cerr << desc << endl;
      return 0; 
    }
    
    cerr << "Bucket size: "<<bucket_size <<endl;
    //cerr << "Data: "<<inputPath <<endl;
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

  StdinStreamReader stream;
  Region universe;

  uint64_t recs = 0 ;
  while (stream.hasNext())
  {
    Data* d = reinterpret_cast<Data*>(stream.getNext());
    if (d == 0)
      throw Tools::IllegalArgumentException(
          "bulkLoadUsingRPLUS: RTree bulk load expects SpatialIndex::RTree::Data entries."
          );
    Region *obj = d->m_region.clone();
    if (0 == recs)
      universe = *obj;
    else 
      universe.combineRegion(*obj);

    delete d;
    delete obj;

    if ((++recs % 5000000) == 0)
      std::cerr << "Procestd::couted records " <<  recs  << std::endl;
  }

  // cerr << "Number of tiles: " << endl;

  // boost::timer t;     
  Timer t;
  genTiles(universe,bucket_size,recs);
  double elapsed_time = t.elapsed();
  cerr<< "stat:ptime," << bucket_size << "," << tiles.size() <<"," << elapsed_time << endl;
  // cleanup allocated memory 
  for (vector<RTree::Data*>::iterator it = tiles.begin() ; it != tiles.end(); ++it) 
  {
    cout << (*it)->m_id << " " << (*it)->m_region <<endl;
    delete *it;
  }

  cout.flush();

  return 0; // success
}

