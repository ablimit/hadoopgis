#include<cmath>
#include <boost/program_options.hpp>
#include "SpaceStreamReader.h"
#include "Timer.hpp"

vector<RTree::Data*> tiles;

// using namespace boost;
namespace po = boost::program_options;
using namespace SpatialIndex::RTree;

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
  string inputPath;

  try {
    po::options_description desc("Options");
    desc.add_options()
        ("help", "this help message")
        ("bucket,b", po::value<uint32_t>(&bucket_size), "Expected bucket size")
        ("input,i", po::value<string>(&inputPath), "Data input file path")
        ;

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

  double low[2], high[2];
  low[0] = 0.0 ;
  low[1] = 0.0 ;
  high[0] = 1.0 ;
  high[1] = 1.0 ;

  Region universe(low,high,2);

  SpaceStreamReader stream(inputPath);
  uint64_t recs = 0 ;
  while (stream.hasNext())
  {
    Data* d = reinterpret_cast<Data*>(stream.getNext());
    if (d == 0)
      throw Tools::IllegalArgumentException(
          "bulkLoadUsingRPLUS: RTree bulk load expects SpatialIndex::RTree::Data entries."
          );
    /*
    Region *obj = d->m_region.clone();
    if (0 == recs)
      universe = *obj;
    else 
      universe.combineRegion(*obj);
    delete obj;
    */
    delete d;

    if ((++recs % 5000000) == 0)
      std::cerr << "Procestd::couted records " <<  recs  << std::endl;
  }

  // cerr << "Number of tiles: " << endl;

  // boost::timer t;     
  Timer t;
  genTiles(universe,bucket_size,recs);
  double elapsed_time = t.elapsed();
  cerr << "stat:ptime," << bucket_size << "," << tiles.size() <<"," << elapsed_time << endl;

  for (vector<RTree::Data*>::iterator it = tiles.begin() ; it != tiles.end(); ++it) 
  {
    cout << (*it)->m_id << " " << (*it)->m_region << endl;
  }

  // cleanup allocated memory 
  for (vector<RTree::Data*>::iterator it = tiles.begin() ; it != tiles.end(); ++it) 
    delete *it;

  cout.flush();

  return 0; // success
}

