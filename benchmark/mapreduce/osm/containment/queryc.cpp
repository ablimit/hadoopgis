#include "hadoopgis.h"
#include "paris.h"

/* local vars  */

const double width  = 0.09;
const double height   = 0.045;
const double origin_x = -180.00 ;
const double origin_y = -90.00 ;
double plow[2], phigh[2];

int counter =0 ;

map<string,string> exact_hits ; 
map<string,string> candidate_hits;

map<string,string> candidate_hits_rec;

GeometryFactory *gf = NULL;
WKTReader *wkt_reader = NULL;
Geometry *paris_poly = NULL; 

/* functions */

string constructBoundary(string tile_id){
    vector<string> strs;
    boost::split(strs, tile_id, boost::is_any_of(UNDERSCORE));
    if (strs.size()<2) {
        cerr << "ERROR: ill formatted tile id." <<endl;
        return "" ;
    }
    stringstream ss;
    int x = boost::lexical_cast< int >( strs[0] );
    int y = boost::lexical_cast< int >( strs[1] );

    // construct a WKT polygon 
    ss << shapebegin ;
    ss << origin_x + (x-1) * width ; ss << SPACE ; ss << origin_y + (y-1) * height ; ss << COMMA;
    ss << origin_x + x     * width ; ss << SPACE ; ss << origin_y + (y-1) * height ; ss << COMMA;
    ss << origin_x + x     * width ; ss << SPACE ; ss << origin_y + y     * height ; ss << COMMA;
    ss << origin_x + (x-1) * width ; ss << SPACE ; ss << origin_y + y     * height ; ss << COMMA;
    ss << origin_x + (x-1) * width ; ss << SPACE ; ss << origin_y + (y-1) * height ; 
    ss << shapeend ;

    return ss.str();
}

int isTileRelevant(bool isTile, string tile_id){
    int val = -1;
    //cerr<< tile_id << TAB << wkt << TAB;
    Geometry *tile_boundary = wkt_reader->read(isTile ? constructBoundary(tile_id): tile_id);
    
    if (paris_poly->intersects(tile_boundary))
    {
	if (paris_poly->contains(tile_boundary))
	    val = 0;
	else 
	    val = 1;
    }

    delete tile_boundary ;
    return val;
}

void processQuery()
{
    cerr << "Started Query Processing...." << endl;
    Geometry * way = NULL; 
    int count =exact_hits.size();
    // polygons which are definitly contained in the boundary
    //for (map<string,string>::iterator it = exact_hits.begin() ; it != exact_hits.end(); ++it)

    // polygons which may be contained in the boundary
    for (map<string,string>::iterator it = candidate_hits.begin() ; it != candidate_hits.end(); ++it)
    {
        way = wkt_reader->read(it->second);

        if (paris_poly->contains(way)) count++;
        delete way;
    }

    cout << "Number of records contained in region: [" <<count <<"]" <<endl;
    cout.flush();
}


int main(int argc, char **argv) {
    string input_line;
    vector<string> fields;
    int rel = -1;

    gf = new GeometryFactory(new PrecisionModel(),PAIS_SRID);
    wkt_reader= new WKTReader(gf);
    paris_poly = wkt_reader->read(paris);
    //cerr << paris_poly->getGeometryType() <<endl; 
    
    while(cin && getline(cin, input_line) && !cin.eof()){
        //cerr << "DEBUG @: " << ++counter << endl;
        //cerr.flush();
        boost::split(fields, input_line, boost::is_any_of(BAR));

        rel= (fields[OSM_TILEID].size()<3) ? isTileRelevant(false,fields[OSM_POLYGON]) : isTileRelevant(true,fields[OSM_TILEID]);

        if (rel == 0)// if tile ID matches, continue searching 
        {
            //cerr << "Exact: "<< fields[OSM_ID] << endl;
            exact_hits[fields[OSM_ID]]=input_line;
            //cout << key<< tab << index<< tab << shapebegin <<value <<shapeend<< endl;
        }
        else if (rel==1)
        {
            //cerr << "Candi: " << fields[OSM_ID] <<endl;
            candidate_hits[fields[OSM_ID]]=fields[OSM_POLYGON];
            candidate_hits_rec[fields[OSM_ID]]=input_line;
        }
        /*
        else {
            cerr << "Empty: " << fields[OSM_ID] <<endl;
        }
        */
        fields.clear();
    }
    //cerr << "okay" << endl;
    cerr<< "Exact hist: " << exact_hits.size() << endl; 
    cerr<< "Candidate: " << candidate_hits.size() << endl; 

    processQuery();

    cerr.flush();
    
    delete paris_poly;
    delete wkt_reader;
    delete gf;

    return 0; // success
}

