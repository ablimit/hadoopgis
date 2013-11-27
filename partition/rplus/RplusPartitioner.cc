#include "./SpaceStreamReader.h"
#include <unordered_map>
#include <cstring>

using namespace SpatialIndex::RTree;

const uint32_t DIM_X = 0;
const uint32_t DIM_Y = 1;
uint64_t TotalEntries = 0;

Region universe;
std::unordered_map<uint64_t,Region*> buffer;

struct SortRegionAscendingX : public std::binary_function<const uint64_t, const uint64_t, bool>
{
  bool operator()(const uint64_t &idx1, const uint64_t &idx2) const 
  {
    if (buffer[idx1]->m_pHigh[0] + buffer[idx1]->m_pLow[0] < buffer[idx2]->m_pHigh[0] + buffer[idx2]->m_pLow[0])
      return true;
    else if (buffer[idx1]->m_pHigh[0] + buffer[idx1]->m_pLow[0] > buffer[idx2]->m_pHigh[0] + buffer[idx2]->m_pLow[0])
      return false;
    else 
      return idx1 < idx2;
  }
};

struct SortRegionAscendingY : public std::binary_function<const uint64_t, const uint64_t, bool>
{
  bool operator()(const uint64_t &idx1, const uint64_t &idx2) const 
  {
    if (buffer[idx1]->m_pHigh[1] + buffer[idx1]->m_pLow[1] < buffer[idx2]->m_pHigh[1] + buffer[idx2]->m_pLow[1])
      return true;
    else if (buffer[idx1]->m_pHigh[1] + buffer[idx1]->m_pLow[1] > buffer[idx2]->m_pHigh[1] + buffer[idx2]->m_pLow[1])
      return false;
    else 
      return idx1 < idx2;
  }
};

std::set<uint64_t, SortRegionAscendingX> xsorted_buffer;
std::set<uint64_t, SortRegionAscendingY> ysorted_buffer;


// functions 
void insert(uint64_t id, Region* r);
void calculateSpatialUniverse();
float getCostX(uint32_t K);
float getCostY(uint32_t K);
Region split_x(uint32_t K, uint32_t & size );
Region split_y(uint32_t K, uint32_t & size );
void cleanup(std::vector<uint64_t> & spatial_objects);

// main method

int main(int argc, char** argv)
{
  if (argc != 3)
  {
    std::cerr << "Usage: " << argv[0] << " [input_file] [partition_size]" << std::endl;
    return 1;
  }

  SpaceStreamReader stream(argv[1]);
  const int partition_size = atoi(argv[2]);
  uint64_t cc = 0; 
  while (stream.hasNext())
  {
    Data* d = reinterpret_cast<Data*>(stream.getNext());
    if (d == 0)
      throw Tools::IllegalArgumentException(
          "bulkLoadUsingRPLUS: RTree bulk load expects SpatialIndex::RTree::Data entries."
          );
    Region *obj = d->m_region.clone();
    insert((uint64_t)d->m_id,obj);
    delete d;
    cc++; 
    if (cc % 100000 == 0)
	std::cerr << "number of records sorted: " << cc << std::endl;
  }

  // initilization 
  TotalEntries = buffer.size();
  universe =  *(buffer.begin()->second);
  calculateSpatialUniverse();

  // summary info 
  std::cerr << "Spatial Universe: " << universe<< std::endl;
  std::cerr << "|spatial objects| = " << TotalEntries
    << ", |X| = " << xsorted_buffer.size() 
    << ", |Y| = " << ysorted_buffer.size() << std::endl;
  std::cerr << "RTree::BulkLoader::R+ packing objects .." << std::endl;

  // loop vars 
  double cost [] = {0.0, 0.0};
  uint64_t iteration = 0; 
  Region pr(universe);
  uint32_t size = 0;
  uint32_t pid = 0;

  while (true)
  {

    cost [0] = 0.0;
    cost [1] = 0.0;
    size = 0 ;
    iteration++;

    if (TotalEntries <= partition_size) {

      pr = split_x(partition_size,size);

      // last partition
      std::cerr << "Iteration: " << iteration << 
        "\tx-cost = " << cost [DIM_X] << 
        "\ty-cost = " << cost [DIM_Y] << 
        "\tRegion = " << pr << std::endl;
      std::cout << ++pid << " " << pr << endl;
      std::cerr << "|objetcs| = " << TotalEntries<< " , |partition| = " << size << std::endl;
      break; 
    }

    cost[DIM_X] = getCostX(partition_size);
    cost[DIM_Y] = getCostY(partition_size);

    pr = (cost[DIM_X] < cost[DIM_Y] )?  split_x(partition_size,size) : split_y(partition_size,size);

    // program state 
    std::cerr << "Iteration: " << iteration << 
      "\tx-cost = " << cost [DIM_X] << 
      "\ty-cost = " << cost [DIM_Y] << 
      "\tRegion = " << pr << std::endl;
      std::cout << ++pid << " " << pr << endl;

    std::cerr << "|objetcs| = " << TotalEntries<< " , |partition| = " << size << std::endl;
  } // end while


  return 0 ;
}

// function implementations 

void calculateSpatialUniverse(){
  for ( std::unordered_map<uint64_t,Region*>::iterator iter = buffer.begin(); iter != buffer.end(); ++iter )
    universe.combineRegion(*(iter->second));
}


void insert(uint64_t id, Region* r)
{
  // check if id is in the collection and report error if duplicates
  std::pair<std::unordered_map<uint64_t, Region*>::iterator, bool> res = buffer.insert(std::make_pair(id,r));
  if ( ! res.second ) {
    std::cerr<< "ERROR: item: " <<  id << " already exists in the map." << std::endl;// " with value " << (res.first)->second << endl;
  }

  // sorted by X Axsis
  std::pair<std::set<uint64_t>::iterator, bool> res_set = xsorted_buffer.insert(id);
  if ( ! res_set.second ) {
    std::cerr<< "ERROR: item: " <<  id << " already exists in the horizontal axis set." << std::endl;// " with value " << (res.first)->second << endl;
  }

  // sorted by Y Axsis
  res_set = ysorted_buffer.insert(id);
  if ( ! res_set.second ) {
    std::cerr<< "ERROR: item: " <<  id << " already exists in the vertical axis set." << std::endl;// " with value " << (res.first)->second << endl;
  }
  TotalEntries++;
}


float getCostX(uint32_t K){
  float cost = 0.0; 

  std::set<uint64_t,SortRegionAscendingX>::iterator iter = xsorted_buffer.begin();
  std::advance(iter,K-1);

  Region * region_k = buffer[*iter];
  double c1 [] = {0.0, 0.0};
  double c2 [] = {0.0, 0.0};

  c1[DIM_X] = region_k->getHigh(DIM_X);
  c1[DIM_Y] = universe.getLow(DIM_Y);
  c2[DIM_X] = c1[DIM_X];
  c2[DIM_Y] = universe.getHigh(DIM_Y);
  const LineSegment lseg(c1,c2,universe.getDimension());

  // iterate on smaller elements
  for ( ; iter != xsorted_buffer.begin(); --iter)
  {
    if (buffer[*iter]->intersectsLineSegment(lseg))
      cost += 1.0 ;
    else 
      break;
  }
  if (iter == xsorted_buffer.begin()) cost += 1.0; 

  // iterate right side 
  iter = xsorted_buffer.begin();
  for (std::advance(iter,K); iter != xsorted_buffer.end(); ++iter)
  {
    if (buffer[*iter]->intersectsLineSegment(lseg))
      cost += 1.0 ;
    else 
      break;
  }

  return cost;
}

float getCostY(uint32_t K){
  float cost = 0.0; 

  std::set<uint64_t,SortRegionAscendingY>::iterator iter = ysorted_buffer.begin();
  std::advance(iter,K-1);

  Region * region_k = buffer[*iter];
  double c1 [] = {0.0, 0.0};
  double c2 [] = {0.0, 0.0};

  c1[DIM_Y] = region_k->getHigh(DIM_Y);
  c1[DIM_X] = universe.getLow(DIM_X);
  c2[DIM_Y] = c1[DIM_Y];
  c2[DIM_X] = universe.getHigh(DIM_X);
  const LineSegment lseg(c1,c2,universe.getDimension());

  // iterate on smaller elements
  for ( ; iter != ysorted_buffer.begin(); --iter)
  {
    if (buffer[*iter]->intersectsLineSegment(lseg))
      cost += 1.0 ;
    else 
      break;
  }
  if (iter == ysorted_buffer.begin()) cost += 1.0; 

  // iterate right side 
  iter = ysorted_buffer.begin();
  for (std::advance(iter,K); iter != ysorted_buffer.end(); ++iter)
  {
    if (buffer[*iter]->intersectsLineSegment(lseg))
      cost += 1.0 ;
    else 
      break;
  }

  return cost;
}

Region split_x(uint32_t K, uint32_t & size )
{
  double c1 [] = {0.0, 0.0};
  double c2 [] = {0.0, 0.0};
  uint32_t dim = DIM_X ; 
  uint32_t adim = DIM_Y ; // another dimension 
  Region p;
  p.makeDimension(2);

  vector<uint64_t> spatial_objects; // object ids which forms a partition

  std::set<uint64_t,SortRegionAscendingX>::iterator iter = xsorted_buffer.begin();

  if ( TotalEntries > K )
  {
    std::advance(iter,K-1);
    Region * region_k = buffer[*iter];
    c1[dim] = region_k->getHigh(dim);
    c1[adim] = universe.getLow(adim);
    c2[dim] = c1[dim];
    c2[adim] = universe.getHigh(adim);

    memcpy(p.m_pLow, universe.m_pLow,   2 * sizeof(double));
    memcpy(p.m_pHigh, c2, 2 * sizeof(double));

    memcpy(universe.m_pLow, c1, 2 * sizeof(double));

  }
  else // if (getTotalEntries() <= K)
  {
    p = universe; 
  }

  // create list of items to clean
  uint32_t count = 0 ; 
  for (iter = xsorted_buffer.begin(); iter != xsorted_buffer.end() && count++ < K  ; ++iter)
    spatial_objects.push_back(*iter);

  // update sorted vectors & free processed objects
  size = spatial_objects.size();
  cleanup(spatial_objects);
  // return the partition
  return p;
}
Region split_y(uint32_t K, uint32_t & size )
{
  double c1 [] = {0.0, 0.0};
  double c2 [] = {0.0, 0.0};
  uint32_t dim = DIM_Y ; 
  uint32_t adim = DIM_X ; // another dimension 
  Region p;
  p.makeDimension(2);

  vector<uint64_t> spatial_objects; // object ids which forms a partition

  std::set<uint64_t,SortRegionAscendingY>::iterator iter = ysorted_buffer.begin();

  if ( TotalEntries > K )
  {
    std::advance(iter,K-1);
    Region * region_k = buffer[*iter];
    c1[dim] = region_k->getHigh(dim);
    c1[adim] = universe.getLow(adim);
    c2[dim] = c1[dim];
    c2[adim] = universe.getHigh(adim);

    memcpy(p.m_pLow, universe.m_pLow,   2 * sizeof(double));
    memcpy(p.m_pHigh, c2, 2 * sizeof(double));

    memcpy(universe.m_pLow, c1, 2 * sizeof(double));

  }
  else // if (getTotalEntries() <= K)
  {
    p = universe; 
  }

  // create list of items to clean
  uint32_t count = 0 ; 
  for (iter = ysorted_buffer.begin(); iter != ysorted_buffer.end() && count++ < K  ; ++iter)
    spatial_objects.push_back(*iter);

  // update sorted vectors & free processed objects
  size = spatial_objects.size();
  cleanup(spatial_objects);

  // return the partition
  return p;
}

void cleanup(std::vector<uint64_t> & spatial_objects)
{
  for (std::vector<uint64_t>::iterator it = spatial_objects.begin(); it != spatial_objects.end(); ++it)
  {
    xsorted_buffer.erase(*it);
    ysorted_buffer.erase(*it);
    delete buffer[*it] ;
    buffer.erase(*it);
  }
  TotalEntries = buffer.size();

}


