#include "hadoopgis.h"
#include <fstream>
#include <sstream>
#include <boost/algorithm/string/join.hpp>

/* local vars  */
double low[2], high[2];


map<int,Envelope*> partition_boundary;

GeometryFactory *gf = NULL;
WKTReader *wkt_reader = NULL;

/* functions */

void constructGeom(string line){
    stringstream ss(line);
    int pid; // partition id

    ss >> pid;
    ss >> low[0] ;
    ss >> low[1] ;
    ss >> high[0] ;
    ss >> high[1] ;

    partition_boundary[pid] = new Envelope(low[0],high[0],low[1],high[1]);
}

int getPartition(string wkt){
    int val = -2;
    //cerr<< tile_id << TAB << wkt << TAB;
    const Envelope * geom = (wkt_reader->read(wkt))->getEnvelopeInternal () ;
    for (map<int,Envelope*>::iterator it = partition_boundary.begin() ; it != partition_boundary.end(); ++it)
    {
        if (it->second->intersects(geom))
        {
            if (it->second->contains(geom))
            {
                val= it->first;
                break;
            }
            else 
                val = -1;
        }

    }

    delete geom;
    return val;
}



int main(int argc, char **argv) {
    if (argc < 2)
    {
        cerr << "Usage: " << argv[0]<< " [mbb] < input"<<endl;
        return -1; 
    }

    ifstream infile(argv[1]);
    string input_line;
    vector<string> fields;
    int p = -1;
    int bobjects=0;

    gf = new GeometryFactory(new PrecisionModel(),PAIS_SRID);
    wkt_reader= new WKTReader(gf);

    while (std::getline(infile, input_line)){ 
        constructGeom(input_line);
    }
    infile.close();

    while(cin && getline(cin, input_line) && !cin.eof()){
        //cerr << "DEBUG @: " << ++counter << endl;
        //cerr.flush();
        boost::split(fields, input_line, boost::is_any_of(BAR));

        p = getPartition(fields[OSM_POLYGON]);

        if (p >=0 )// find a partition 
        {
            ostringstream convert;   // stream used for the conversion
            convert << p;
            fields[OSM_TILEID] = convert.str(); 
            cout << boost::algorithm::join(fields, BAR)<<endl;

        }
        else if (p == -1)
        {
            bobjects++;
            //cerr << "Candi: " << fields[OSM_ID] <<endl;
        }
	else {
	    cerr << "what the hell is this ? " << endl;
	}
        fields.clear();
    }

    cerr << "Number of objects on the partition boundary: " << bobjects << endl;
    cerr.flush();

    delete wkt_reader;
    delete gf;

    return 0; // success
}

