#include "hadoopgis.h"
#include "vecstream.h"

const string tileID = "1985_3049";
const string region ="POLYGON((-1.43999 47.16,-1.07999 47.16,-1.07999 47.34,-1.43999 47.34,-1.43999 47.16))" ;

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

    cout.flush();
}

bool paisUIDMatch(string pais_uid)
{
    char * filename = getenv("map_input_file");
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
    vector<string> fields;

    while(cin && getline(cin, input_line) && !cin.eof()){
        boost::split(fields, input_line, boost::is_any_of(BAR));

        if (fields[OSM_TILEID].size()> 0 && fields[OSM_TILEID].compare(tileID)==0) // if tile ID matches, continue searching 
        {
            geometry_collction.push_back(fields[OSM_POLYGON]);
            //cout << key<< tab << index<< tab << shapebegin <<value <<shapeend<< endl;
        }
        fields.clear();
    }

    processQuery();

    return 0; // success
}

