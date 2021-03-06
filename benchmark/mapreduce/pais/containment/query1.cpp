#include "hadoopgis.h"
#include "vecstream.h"

const string paisUID = "gbm1.1";
const string tileID = "gbm1.1-0000040960-0000040960";
const string region ="POLYGON((40960 40960, 41984 40960,  41984 41984, 40960 41984, 40960 40960))" ;
vector<string> geometry_collction ; 
double plow[2], phigh[2];


void processQuery()
{
    if (geometry_collction.size()>0)
    {
	id_type  indexIdentifier ;
	IStorageManager * storage = StorageManager::createNewMemoryStorageManager();
	ContainmentDataStream stream(&geometry_collction);
	ISpatialIndex * spidx = RTree::createAndBulkLoadNewRTree(RTree::BLM_STR, stream, *storage, 
		FillFactor, IndexCapacity, LeafCapacity, 2, 
		RTree::RV_RSTAR, indexIdentifier);

	// Error checking 
	bool ret = spidx->isIndexValid();
	if (ret == false) std::cerr << "ERROR: Structure is invalid!" << std::endl;
	// else std::cerr << "The stucture seems O.K." << std::endl;
	polygon container ; 
	box container_mbb;
	boost::geometry::read_wkt(region, container);
	boost::geometry::envelope(container,container_mbb);
	plow [0] = boost::geometry::get<boost::geometry::min_corner, 0>(container_mbb);
	plow [1] = boost::geometry::get<boost::geometry::min_corner, 1>(container_mbb);

	phigh [0] = boost::geometry::get<boost::geometry::max_corner, 0>(container_mbb);
	phigh [1] = boost::geometry::get<boost::geometry::max_corner, 1>(container_mbb);

	Region r = Region(plow, phigh, 2);
	MyVisitor vis ; 
	spidx->containsWhatQuery(r, vis);


	// garbage collection 
	delete spidx;
	delete storage;
    }
}

bool paisUIDMatch(string pais_uid)
{
    char * filename = getenv("map_input_file");
    //char * filename = "astroII.1.1";
    if ( NULL == filename ){
	cerr << "map.input.file is NULL." << endl;
	return false;
    }

    if (pais_uid.compare(filename) ==0)
	return true;
    else 
	return false;
}

int main(int argc, char **argv) {
    string input_line;
    string tile_id ;
    string poly;

    if (paisUIDMatch(paisUID))
    {
	while(cin && getline(cin, input_line) && !cin.eof()){

	    size_t pos = input_line.find_first_of(comma,0);
	    if (pos == string::npos)
		return 1; // failure

	    tile_id = input_line.substr(0,pos);
	    if (tileID.compare(tile_id)==0) // if tile ID matches, continue searching 
	    {
		pos=input_line.find_first_of(comma,pos+1);
		geometry_collction.push_back(shapebegin + input_line.substr(pos+2,input_line.length()- pos - 3) + shapeend);
		//cout << key<< tab << index<< tab << shapebegin <<value <<shapeend<< endl;
	    }
	}
    }

    processQuery();

    cout.flush();
    return 0; // success
}

