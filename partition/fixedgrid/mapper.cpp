#include "hadoopgis.h"
#include "cmdline.h"

/*
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
*/
vector<string> parsePAIS(string & line) {
    vector<string> vec ;
	    size_t pos = line.find_first_of(comma,0);
	    size_t pos2;
	    if (pos == string::npos){
            return vec; // failure
        }

	    vec.push_back(line.substr(0,pos)); // tile_id
		pos2=line.find_first_of(comma,pos+1);
		vec.push_back(line.substr(pos+1,pos2-pos-1)); // object_id
		pos=pos2;
		vec.push_back(shapebegin + line.substr(pos+2,line.length()- pos - 3) + shapeend);
}

int main(int argc, char **argv) {
    gengetopt_args_info args_info;
    if (cmdline_parser (argc, argv, &args_info) != 0)
        exit(1) ;

    int min_x = args_info.min_x_given;
    int max_x = args_info.max_x_given;
    int min_y = args_info.min_y_given;
    int max_y = args_info.max_y_given;
    int x_split = args_info.x_split_given;
    int y_split = args_info.y_split_given;


    string input_line;
    vector<string> fields;
    cerr << "Reading input from stdin..." <<endl;
    int c = 0 ;
    while(cin && getline(cin, input_line) && !cin.eof()){
        fields = parsePAIS(input_line);

    }

    cout.flush();
    cerr.flush();
    cmdline_parser_free (&args_info); /* release allocated memory */
    return 0; // success
}


