#include "../../SpaceStreamReader.h"
#include<cmath>
#include <boost/timer.hpp>
#include <boost/program_options.hpp>

vector<RTree::Data*> tiles;

// using namespace boost;
namespace po = boost::program_options;
using namespace SpatialIndex::RTree;

void genTiles(Region &universe, const uint32_t partition_size, const uint64_t ds) {
  double P = std::ceil(static_cast<double>(ds) / static_cast<double>(partition_size));
  double split  =std::ceil(std::sqrt(P));
  uint64_t LOOP =static_cast<uint64_t>(std::ceil(std::sqrt(P)));

  cerr << "ds: " << ds << ", partition size:  "<< partition_size << endl;
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
  uint32_t partition_size ;
  int indexCapacity ;
  int leafCapacity ;
  double fillFactor ;
  try {
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help", "produce help message")
        ("index_cap", po::value<int>(&indexCapacity)->default_value(20), "RTree index page size")
        ("leaf_cap", po::value<int>(&leafCapacity)->default_value(1000), "RTree leaf page size")
        ("fill_cap", po::value<double>(&fillFactor)->default_value(0.99), "Page fill factor.")
        ("bucket_size", po::value<int>(), "Expected bucket size")
        ;

    po::variables_map vm;        
    po::store(po::parse_command_line(ac, av, desc), vm);
    po::notify(vm);    

    if (vm.count("help")) {
      cerr << desc << endl;
      return 0;
    }

    if (vm.count("fill_cap")) {
      cout << "fill factor was set to " 
          << vm["fill_cap"].as<double>() << ".\n";
    } else {
      cout << "fill factor was not set.\n";
    }

    return 0; 
  }
  catch(exception& e) {
    cerr << "error: " << e.what() << "\n";
    return 1;
  }
  catch(...) {
    cerr << "Exception of unknown type!\n";
    return 1;
  }

  SpaceStreamReader stream("");
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

  boost::timer t;                         
  genTiles(universe,partition_size,recs);
  double elapsed_time = t.elapsed();
  cerr << "stat:" << partition_size << "," << tiles.size() <<"," << elapsed_time << endl;

  // build in memory Tree 
  t.restart();


  elapsed_time = t.elapsed();
  // assign partition id 




  return 0; // success
}

