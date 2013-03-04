#include "hadoopgis.h"

/* local vars  */
const string tileID = "1985_3049";
vector<string> exact_hits ; 

GeometryFactory *gf = NULL;
WKTReader *wkt_reader = NULL;

/* functions */
void processQuery()
{
    Geometry * way = NULL; 
    for (vector<string>::iterator it = exact_hits.begin() ; it != exact_hits.end(); ++it)
    {
	way = wkt_reader->read(*it);
	cout << way->getArea() << TAB 
	     << way->getCentroid()->toString() << TAB
	     << way->convexHull()->toString()  << TAB
	     << way->getLength() << endl;
	delete way;
    }
    cout.flush();
}


int main(int argc, char **argv) {
    string input_line;
    vector<string> fields;

    gf = new GeometryFactory(new PrecisionModel(),OSM_SRID);
    wkt_reader= new WKTReader(gf);


    while(cin && getline(cin, input_line) && !cin.eof()){
	boost::split(fields, input_line, boost::is_any_of(BAR));

        if (fields[OSM_TILEID].size()> 0 && fields[OSM_TILEID].compare(tileID)==0) // if tile ID matches, continue searching 
        {
            exact_hits.push_back(fields[OSM_POLYGON]);
            //cout << key<< tab << index<< tab << shapebegin <<value <<shapeend<< endl;
        }
	fields.clear();
    }

    processQuery();

    delete wkt_reader;
    delete gf;

    return 0; // success
}

