#include "hadoopgis.h"
#include "vecstream.h"

const int tile_size  = 4096;
const float AVG_AREA = 125.0;
const string paisUID = "gbm1.1";
const string region ="POLYGON((22528 8192,67584 8192,67584 24576,22528 24576,22528 8192))";
polygon poly;
map<string,bool> tileIDSet;
vector<string> geometry_collction ; 
double plow[2], phigh[2];

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
    if (tileIDSet.find(tile_id)== tileIDSet.end())
    {
    string wkt= constructBoundary(tile_id);
    polygon poly1; 
    boost::geometry::read_wkt(wkt, poly1);
    bool res = boost::geometry::intersects(poly,poly1);
    tileIDSet[tile_id] = res;
    }

    return tileIDSet[tile_id] ;
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


int main(int argc, char **argv) {
    string input_line;
    string tile_id ;
    polygon polygon_object ;

    boost::geometry::read_wkt(region, poly);

    if (paisUIDMatch(paisUID))
    {
        while(cin && getline(cin, input_line) && !cin.eof()){

            size_t pos = input_line.find_first_of(comma,0);
            if (pos == string::npos)
                return 1; // failure

            tile_id = input_line.substr(0,pos);
            if (isTileRelevant(tile_id)) // if tile ID matches, continue searching 
            {
                pos=input_line.find_first_of(comma,pos+1);
                string sobject = shapebegin + input_line.substr(pos+2,input_line.length()- pos - 3) + shapeend;
                boost::geometry::read_wkt(sobject, polygon_object);  // spatial filtering
                if (boost::geometry::area(polygon_object)>AVG_AREA)
                    geometry_collction.push_back(sobject);
                //cout << key<< tab << index << tab << shapebegin <<value <<shapeend<< endl;
            }
        }
    }

    processQuery();

    cout.flush();
    return 0; // success
}

