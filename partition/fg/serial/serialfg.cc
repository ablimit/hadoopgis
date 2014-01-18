#include "../../SpaceStreamReader.h"
#include<cmath>
#include <boost/timer.hpp>>

vector<RTree::Data*> tiles;

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
      tiles.push_back(new RTree::Data(0, 0 , r, tid++);
    }
  }
}

int main(int argc, char **argv) {
  if (argc < 3)
  {
    std::cerr << "Usage: " << argv[0] << " [input_file] [partition_size]" << std::endl;
    return 1;
  }

  SpaceStreamReader stream(argv[1]);
  const uint32_t partition_size = strtoul (argv[2], NULL, 0);
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

