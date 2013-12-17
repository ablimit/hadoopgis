#include "./SpaceStreamReader.h"
#include<cmath>

using namespace SpatialIndex::RTree;

uint32_t genTiles(Region &universe, const uint32_t partition_size, const uint64_t ds) {
    vector<Region*> tiles;
    double P = std::ceil(static_cast<double>(ds) / static_cast<double>(partition_size));
    double split  =std::ceil(std::sqrt(P));
    uint64_t LOOP =static_cast<uint64_t>(std::ceil(std::sqrt(P)));
    
    double width =  (universe.getHigh(0)- universe.getLow(0)) / split;
    //cerr << "Tile width" << SPACE <<width <<endl;
    double height = (universe.getHigh(1)- universe.getLow(1)) / split;
    //cerr << "Tile height" << SPACE << height <<endl;

    double min_x = universe.getLow(0);
    double min_y = universe.getLow(1);
    string SPACE = " ";
    uint32_t tid = 0 ; 
    for (uint32_t i =0 ; i < LOOP ; i++)
    {
	for (uint32_t j =0 ; j< LOOP; j++)
	{
	    std::cout << tid << SPACE;
	    std::cout << min_x + i * width << SPACE 
                      << min_y+ j * height << SPACE 
                      << min_x + (i+1) * width << SPACE 
                      << min_y + (j+1) * height << std::endl;
            tid++;
	}
    }

    return tid;
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
    
    cerr << "Number of tiles: " << endl;

    uint32_t size = genTiles(universe,partition_size,recs);
    cerr << "Number of tiles: " << size << endl;
    return 0; // success
}

