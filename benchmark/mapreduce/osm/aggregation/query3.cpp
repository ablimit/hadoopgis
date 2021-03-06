#include "hadoopgis.h"

/* local vars  */
const double AREA_PREDICATE = 0.1;
vector<string> exact_hits ;

GeometryFactory *gf = NULL;
WKTReader *wkt_reader = NULL;

/* functions */
void processQuery() { cout.flush(); }

int main(int argc, char **argv) {
    string input_line;
    vector<string> fields;
    Geometry * way = NULL; 

    gf = new GeometryFactory(new PrecisionModel(),OSM_SRID);
    wkt_reader= new WKTReader(gf);

    while(cin && getline(cin, input_line) && !cin.eof()){

	boost::split(fields, input_line, boost::is_any_of(BAR));
	way = wkt_reader->read(fields[OSM_POLYGON]);
	if (way->getArea()> AREA_PREDICATE) {
	    cout << way->getArea() << TAB 
		<< way->getCentroid()->toString() << TAB
		<< way->convexHull()->toString()  << TAB
		<< way->getLength() << endl;
	}
	delete way;
	fields.clear();
    }

    processQuery();

    delete wkt_reader;
    delete gf;

    return 0; // success
}

