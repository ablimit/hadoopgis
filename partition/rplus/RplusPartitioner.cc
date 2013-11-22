#include "SpaceStreamReader.h"
#include <unordered_map>
// #include <functional>

/*
#include <cstring>
#include <stdio.h>
#include <cmath>
#include "RTree.h"
#include "Leaf.h"
#include "Index.h"
#include "BulkLoader.h"
*/
using namespace SpatialIndex::RTree;

Region universe;		

uint64_t TotalEntries = 0;
uint64_t pos = -1;
uint32_t last_dim =-1; 

std::unordered_map<uint64_t,Region*> buffer;

struct SortRegionAscendingX : public std::binary_function<const uint64_t, const uint64_t, bool>
{
    bool operator()(const uint64_t &idx1, const uint64_t &idx2) const 
    {

	if (buffer[idx1]->m_pHigh[0] + buffer[idx1]->m_pLow[0] < buffer[idx2]->m_pHigh[0] + buffer[idx2]->m_pLow[0])
	    return true;
	else
	    return false;
    }
};
struct SortRegionAscendingY : public std::binary_function<const uint64_t, const uint64_t, bool>
{
    bool operator()(const uint64_t &idx1, const uint64_t &idx2) const 
    {

	if (buffer[idx1]->m_pHigh[1] + buffer[idx1]->m_pLow[1] < buffer[idx2]->m_pHigh[1] + buffer[idx2]->m_pLow[1])
	    return true;
	else
	    return false;
    }
};

std::set<uint64_t, SortRegionAscendingX> xsorted_buffer;
std::set<uint64_t, SortRegionAscendingY> ysorted_buffer;

void insert(uint64_t id, Region* r)
{
    // check if id is in the collection and report error if duplicates
    std::pair<std::unordered_map<uint64_t, Region*>::iterator, bool> res = buffer.insert(std::make_pair(id,r));
    if ( ! res.second ) {
	std::cerr<< "ERROR: item: " <<  id << " already exists in the collection." << std::endl;// " with value " << (res.first)->second << endl;
    }

    // sorted by X Axsis
    std::pair<std::set<uint64_t>::iterator, bool> res_set = xsorted_buffer.insert(id);
    if ( ! res_set.second ) {
	std::cerr<< "ERROR: item: " <<  id << " already exists in the collection." << std::endl;// " with value " << (res.first)->second << endl;
    }

    // sorted by Y Axsis
    res_set = ysorted_buffer.insert(id);
    if ( ! res_set.second ) {
	std::cerr<< "ERROR: item: " <<  id << " already exists in the collection." << std::endl;// " with value " << (res.first)->second << endl;
    }
}

/*
void insert(uint64_t id, Region* r);
void insert(Record* r);
void sort();
void sort(uint32_t dim, uint64_t K=0);
Region getUniverse();
float getCost(uint32_t K, uint32_t dim);
//Region getRegion(uint32_t K);
Record* getNextRecord();
void split(uint32_t K,uint32_t dim, Region &r, std::vector<Record*> &node);
*/

int main(int argc, char** argv)
{
    if (argc != 3)
    {
	std::cerr << "Usage: " << argv[0] << " [input_file] [partition_size]" << std::endl;
	return 1;
    }

    SpaceStreamReader stream(argv[1]);
    const int partition_size = atoi(argv[2]);

    while (stream.hasNext())
    {
	Data* d = reinterpret_cast<Data*>(stream.getNext());
	if (d == 0)
	    throw Tools::IllegalArgumentException(
		    "bulkLoadUsingRPLUS: RTree bulk load expects SpatialIndex::RTree::Data entries."
		    );
	Region *obj = d->m_region.clone();
	insert(d->m_id,obj);
	delete d;
    }

    // Region r = es->getUniverse();

    // TODO 
    //





    return 0 ;
}

