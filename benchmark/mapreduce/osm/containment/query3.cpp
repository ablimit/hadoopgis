#include "hadoopgis.h"
#include "england.h"

/* local vars  */

const double width  = 0.09;
const double height   = 0.045;
const double origin_x = -180.00 ;
const double origin_y = -90.00 ;
double plow[2], phigh[2];
    
vector<string> exact_hits ; 
vector<string> candidate_hits;

GeometryFactory *gf = NULL;
WKTReader *wkt_reader = NULL;
Geometry *england_poly = NULL; 

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

int isTileRelevant(string tile_id){
    if (tile_id.size()<3)
	return 2;

    int val = -1; 
    string wkt = constructBoundary(tile_id);

    Geometry *tile_boundary = wkt_reader->read(wkt);
    
    if (england_poly->intersects(tile_boundary))
    {
	if (england_poly->contains(tile_boundary))
	    val = 0;
	else 
	    val = 1;
    }
    val =2;

    delete tile_boundary ;
    return val;
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
    string input_line;
    vector<string> fields;

    gf = new GeometryFactory(new PrecisionModel(),PAIS_SRID);
    wkt_reader= new WKTReader(gf);
    england_poly = wkt_reader->read(england);
    
    while(cin && getline(cin, input_line) && !cin.eof()){
	boost::split(fields, input_line, boost::is_any_of(BAR));

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

