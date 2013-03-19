#include "hadoopgis.h"

#include <fstream>
#include <sstream>

#include <boost/algorithm/string/join.hpp>

/* local vars  */

double plow[2], phigh[2];

map<string,string> tilePartitionMap;
map<string,vector<Envelope> > splitTileMap;

GeometryFactory *gf = NULL;
WKTReader *wkt_reader = NULL;

/* functions */

void processSplitTile(string tile_id, string coordinates){
    vector<string> fields;
    string tid;

    boost::split(fields, tile_id, boost::is_any_of(UNDERSCORE));
    fields.pop_back();
    tid = boost::algorithm::join(fields, "_");
    fields.clear();

    boost::split(fields, coordinates, boost::is_any_of(SPACE));

    double low[2];
    double high[2] ;
    low[0] =boost::lexical_cast< double >( fields[0] );
    low[1] =boost::lexical_cast< double >( fields[1] );
    low[0] =boost::lexical_cast< double >( fields[2] );
    low[0] =boost::lexical_cast< double >( fields[3] );

    Envelope mbb(low[0],high[0],low[1],high[1]);

    splitTileMap[tid].push_back(mbb);
}

string getSplitTileId(string tile_id, string way)
{
    int idx = 0; 
    stringstream ss;
    Geometry * geom= wkt_reader->read(way); 

    vector<Envelope> regions =  splitTileMap[tile_id];
    // polygons which are definitly contained in the boundary
    for (vector<Envelope>::iterator it = regions.begin() ; it != regions.end(); ++it)
    {
        if (it->contains(geom->getEnvelopeInternal()))
            break;
        ++idx;
    }
    if (idx<4)
        ss << tile_id << idx ;
    else 
        ss << "NULL";

    return ss.str();
}


int main(int argc, char **argv) {
    if (argc < 2)
    {
        cerr << "Usage: " << argv[0]<< " [data2] < input"<<endl;
        return -1; 
    }

    ifstream infile(argv[1]);

    string input_line;
    vector<string> fields;
    vector<string> sp;

    gf = new GeometryFactory(new PrecisionModel(),PAIS_SRID);
    wkt_reader= new WKTReader(gf);

    string tile_id ;
    string part_id;
    // index data from Kai
    while (std::getline(infile, input_line))
    {
        boost::split(fields, input_line, boost::is_any_of(TAB));
        tile_id = fields[1];

        if (tilePartitionMap.count(tile_id) == 0)
            tilePartitionMap[tile_id]=fields[0]; // TILE_ID ---> Partition_ID

        if (std::count(tile_id.begin(), tile_id.end(), '_')>1)  // the ones got split 
            processSplitTile(tile_id, sp[2]);
        
        fields.clear();
    }

    // osm europe data 
    while(cin && getline(cin, input_line) && !cin.eof()){
        boost::split(fields, input_line, boost::is_any_of(BAR));
        if (fields[OSM_TILEID].size()>2)  // if this object is not on the boundary 
        {
            if (tilePartitionMap.count(fields[OSM_TILEID])>0) // the tile is in the partition
            {
                tile_id = fields[OSM_TILEID] ;
            }
            else if (splitTileMap.count(fields[OSM_TILEID])>0) // the tile is split
            {
                tile_id = getSplitTileId(fields[OSM_TILEID], fields[OSM_POLYGON]);
            }
            else {
                cerr << "ERROR: [" << fields[OSM_TILEID] << "]" <<endl;
            }

            if (tile_id.compare("NULL") !=0 ){
                part_id = tilePartitionMap[tile_id];
                cout << part_id << TAB << tile_id << TAB << fields[OSM_ID] << TAB << fields[OSM_POLYGON] <<endl;
            }
        }
        fields.clear();
    }

    delete wkt_reader;
    delete gf;

    return 0; // success
}

