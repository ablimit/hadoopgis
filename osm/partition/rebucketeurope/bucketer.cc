#include "hadoopgis.h"

#include <fstream>
#include <sstream>

#include <boost/algorithm/string/join.hpp>

/* local vars  */

double plow[2], phigh[2];

map<string,string> tilePartitionMap;
map<string,vector<Envlope> > splitTileMap;

GeometryFactory *gf = NULL;
WKTReader *wkt_reader = NULL;

/* functions */

string constructBoundary(string tile_id){
    vector<string> strs;
    boost::split(strs, tile_id, boost::is_any_of(UNDERSCORE));
    if (strs.size()<2) {
	cerr << "ERROR: ill formatted tile id." <<endl;
	return "" ;
    }
    stringstream ss;
    int x = boost::lexical_cast< int >( strs[1] );
    int y = boost::lexical_cast< int >( strs[2] );

    // construct a WKT polygon 
    ss << shapebegin ;
    ss << origin_x + (x-1) * width ; ss << SPACE ; ss << origin_y + (y-1) * height ; ss << COMMA;
    ss << origin_x + x     * width ; ss << SPACE ; ss << origin_y + (y-1) * height ; ss << COMMA;
    ss << origin_x + x     * width ; ss << SPACE ; ss << origin_y + y     * height ; ss << COMMA;
    ss << origin_x + (x-1) * width ; ss << SPACE ; ss << origin_y + y     * height ; ss << COMMA;
    ss << origin_x + (x-1) * width ; ss << SPACE ; ss << origin_y + (y-1) * height ; ss << COMMA;
    ss << shapeend ;

    return ss.str();
}

void processSplitTile(string tile_id, string coordinates){
    int val = -1; 
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

    string wkt = constructBoundary(tile_id);

    Envlope mbb(low[0],high[0],low[1],high[1]);

    splitTileMap[tid].push_back(mbb);
}

void processQuery()
{
    Geometry * way = NULL; 
    // polygons which are definitly contained in the boundary
    for (vector<string>::iterator it = exact_hits.begin() ; it != exact_hits.end(); ++it)
	cout << *it<< endl;

    // polygons which may be contained in the boundary
    for (vector<string>::iterator it = candidate_hits.begin() ; it != candidate_hits.end(); ++it)
    {
	way = wkt_reader->read(*it);

	if (england_poly->contains(way))
	    cout << *it <<endl;
	delete way;
    }
    cout.flush();
}


int main(int argc, char **argv) {
    if (argc < 2)
    {
	cerr << "Usage: [data2] < input"<<endl;
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
    while (std::getline(infile, input_line))
    {
	boost::split(fields, input_line, boost::is_any_of(TAB));
	tile_id = fields[1];

	if (tile_id.size()<3)
	    continue; // boundary objects

	if (tilePartitionMap.count(tile_id) == 0)
	    tilePartitionMap[tile_id]=fields[0]; //file Mapping 
	if (std::count(tile_id.begin(), tile_id.end(), '_')>1)
	    processSplitTile(tile_id, sp[2]);
	fields.clear();
    }
    
    // osm europe data 
    while(cin && getline(cin, input_line) && !cin.eof()){
	boost::split(fields, input_line, boost::is_any_of(BAR));

	if (tilePartitionMap.count(fields[OSM_TILEID])>0) // the tile is in the partition
	{

	}
	else {
	}

	int rel =isTileRelevant(fields[OSM_TILEID]);
	if (rel == 0)// if tile ID matches, continue searching 
	{
	    exact_hits.push_back(fields[OSM_POLYGON]);
	    //cout << key<< tab << index<< tab << shapebegin <<value <<shapeend<< endl;
	}
	else if (rel==1)
	{
	    candidate_hits.push_back(fields[OSM_POLYGON]);
	}
	fields.clear();
    }

    processQuery();

    delete england_poly;
    delete wkt_reader;
    delete gf;

    return 0; // success
}

