#include "hadoopgis.h"

/* local vars  */
vector<string> exact_hits ;

GeometryFactory *gf = NULL;
WKTReader *wkt_reader = NULL;
double avg_area = 0.0;
double avg_perimeter = 0.0;
int size= 0;

/* functions */
void processQuery() { 
    cout << "K" << TAB <<avg_area<< TAB << avg_perimeter<< TAB << size<<endl;
    cout.flush(); 
}

int main(int argc, char **argv) {
    string input_line;
    vector<string> fields;
    Geometry * way = NULL; 

    gf = new GeometryFactory(new PrecisionModel(),OSM_SRID);
    wkt_reader= new WKTReader(gf);

    while(cin && getline(cin, input_line) && !cin.eof()){

        boost::split(fields, input_line, boost::is_any_of(BAR));
        way = wkt_reader->read(fields[OSM_POLYGON]);

        avg_area += way->getArea();
        avg_perimeter += way->getLength();
        size++; 
        
        delete way;
        fields.clear();
    }


    processQuery();

    delete wkt_reader;
    delete gf;

    return 0; // success
}

