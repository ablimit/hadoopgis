#include "MReducerx.h"
#include <boost/progress.hpp>

bool parseSpatialInput() {
    boost::timer t;                         
    string input_line;
    int index =-1;
    string key ;
    string value;
    size_t pos;
    double * corner =NULL; 

    GeometryFactory *gf = new GeometryFactory(new PrecisionModel(),OSM_SRID);
    WKTReader *wkt_reader= new WKTReader(gf);
    Geometry *poly = NULL; 
    int cc =0 ; 

    //boost::progress_timer t;  // start timing
    
    while(cin && getline(cin, input_line) && !cin.eof()) {
	pos = input_line.find_first_of(tab);
	key = input_line.substr(0,pos);
	index = input_line[pos+1] - '1';   // the very first character denotes join index
	pos = input_line.find_first_of(tab, pos+1);
	value = input_line.substr(pos+1); 
	poly = wkt_reader->read(value);

	const Envelope * env = poly->getEnvelopeInternal();

	corner = new double[4];
	corner[0] = env->getMinX();
	corner[1] = env->getMinY();
	corner[2] = env->getMaxX();
	corner[3] = env->getMaxY();

	cornerdata[index].push_back(corner);

	polydata[index].push_back(poly);
    }

    double elapsed_time = t.elapsed();

    //t.restart();                      
    cerr << "Reading + Parsing cost:" << elapsed_time << endl;
    cerr << "join size" << "\t" << polydata.size() << endl;
    cerr << "Record size [0] --" << "\t" << polydata[0].size() << endl;
    cerr << "Record size [1] --" << "\t" << polydata[1].size() << endl;
    cerr.flush();
    return true;
}


int main(int argc, char** argv)
{
    boost::timer t;
    
    IStorageManager *storages_0 = NULL ;//new IStorageManager * [data.size()];
    IStorageManager *storages_1 = NULL ;//new IStorageManager * [data.size()];
    ISpatialIndex ** forest = new ISpatialIndex * [2];
    id_type indexIdentifier_0;
    id_type indexIdentifier_1;

    bool initilize = true;
    string key ;
    int size = -1;
    bool skip= false;
    int joincar = 0;

	double elapsed_time = t.elapsed();
    
	if (! parseSpatialInput()) 
    {
	cerr << "Reduce input parsing error...." << endl;
	return 1;
    }



    try { 
	// start timing 
	t.restart();
	storages_0= StorageManager::createNewMemoryStorageManager();
	storages_1= StorageManager::createNewMemoryStorageManager();

	MRJDataStream stream_0(&(cornerdata[0]),1);
	MRJDataStream stream_1(&(cornerdata[1]),2);

	// Create and bulk load a new RTree with dimensionality 2, 
	// using memory as the StorageManager and the RSTAR splitting policy.
	forest[0] = RTree::createAndBulkLoadNewRTree( 
		RTree::BLM_STR, stream_0, *storages_0, FillFactor, IndexCapacity, LeafCapacity, 2, 
		SpatialIndex::RTree::RV_RSTAR, indexIdentifier_0);

	forest[1]= RTree::createAndBulkLoadNewRTree( 
		RTree::BLM_STR, stream_1, *storages_1, FillFactor, IndexCapacity, LeafCapacity, 2, 
		SpatialIndex::RTree::RV_RSTAR, indexIdentifier_1);

	bool ret = forest[0]->isIndexValid() && forest[1]->isIndexValid() ;

	// end timing 
	elapsed_time = t.elapsed();
	
	if (ret == false) 
	    cerr << "ERROR: Structure is invalid!" << endl;
	cerr << "RTree constrcution cost: " << elapsed_time<< endl;

	// start timing 
	t.restart();
	
	MRJVisitor vis;
	forest[0]->joinMQuery(vis, forest, 2);
	elapsed_time = t.elapsed();
	cerr << "Filter cost: " << elapsed_time << endl;
	// end timing 

	// start timing 
	t.restart();
	vis.getInfo();
	elapsed_time = t.elapsed();
	cerr << "Refine cost: " << elapsed_time << endl;
	// end timing     
    }
    catch (Tools::Exception& e)
    {
	std::cerr << "******ERROR******" << std::endl;
	std::string s = e.what();
	std::cerr << s << std::endl;
	return -1;
    }

    return 0;
}

