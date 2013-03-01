#include "hadoopgis.h"
#include "vecstream.h"

const int tile_size  = 4096;
const string region ="POLYGON((40960 40960, 40960 41984, 41984 41984, 41984 40960, 40960 40960))" ;

vector<string> geometry_collction ; 
double plow[2], phigh[2];
polygon poly;

string constructBoundary(string tile_id){
    vector<string> strs;
    boost::split(strs, tile_id, boost::is_any_of("-"));
    if (strs.size()<3 ) {
        cerr << "ERROR: Ill formatted tile id." <<endl;
        return "" ;
    }
    stringstream ss;
    int x = boost::lexical_cast< int >( strs[1] );
    int y = boost::lexical_cast< int >( strs[2] );

    // construct a WKT polygon 
    ss << shapebegin ;
    ss << x ;             ss << space ; ss << y ;             ss << comma;
    ss << x ;             ss << space ; ss << y + TILE_SIZE ; ss << comma;
    ss << x + TILE_SIZE ; ss << space ; ss << y + TILE_SIZE ; ss << comma;
    ss << x + TILE_SIZE ; ss << space ; ss << y ;             ss << comma;
    ss << x ;             ss << space ; ss << y ;             ss << comma;
    ss << shapeend ;

    return ss.str();
}

bool isTileRelevant(string tile_id){
    string wkt= constructBoundary(tile_id);
    polygon poly1; 
    boost::geometry::read_wkt(wkt, poly1);
    return boost::geometry::intersects(poly,poly1); 
}

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
        box container_mbb;
        boost::geometry::envelope(poly,container_mbb);
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


int main(int argc, char **argv) {
    string input_line;
    string tile_id ;
    string oid ;

    boost::geometry::read_wkt(region, poly);

    while(cin && getline(cin, input_line) && !cin.eof()){

        size_t pos = input_line.find_first_of(comma,0);
        if (pos == string::npos)
            return 1; // failure

        tile_id = input_line.substr(0,pos);
        if (isTileRelevant(tile_id)) // if tile ID matches, continue searching 
        {
            pos=input_line.find_first_of(comma,pos+1);
            geometry_collction.push_back(shapebegin + input_line.substr(pos+2,input_line.length()- pos - 3) + shapeend);
            //geometry_collction.push_back(input_line.substr(pos+1,string::npos));
            //cout << key<< tab << index<< tab << shapebegin <<value <<shapeend<< endl;
        }
    }
    //cerr << "Number of objects contained in the candidate list: " << geometry_collction.size() << endl;
    //cerr.flush();
    processQuery();

    cout.flush();
    return 0; // success
}

