#include <string.h>

// geos 
#include <geos/geom/PrecisionModel.h>
#include <geos/geom/GeometryFactory.h>
#include <geos/geom/Geometry.h>
#include <geos/geom/Point.h>
#include <geos/io/WKTReader.h>

// boost 
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>

using namespace std;
using namespace geos;
using namespace geos::io;
using namespace geos::geom;

/*reads in the PAIS data and generates MBB for spatial objects. */

int main(int argc, char** argv)
{
    GeometryFactory *gf = new GeometryFactory(new PrecisionModel(),4326);
    WKTReader *wkt_reader= new WKTReader(gf);
    Geometry *poly ; 
    string DEL = " ";
    vector<string> fields;
    double low[2], high[2];
    string input_line ;
    while(cin && getline(cin, input_line) && !cin.eof())
    {
        boost::split(fields, input_line, boost::is_any_of("|"));
        poly = wkt_reader->read(fields[2]);
        const Envelope * env = poly->getEnvelopeInternal();
        low [0] = env->getMinX();
        low [1] = env->getMinY();
        high [0] = env->getMaxX();
        high [1] = env->getMaxY();
        int id = boost::lexical_cast< int >(fields[1]);
        cout << id << DEL << low[0] << DEL << low[1] << DEL << high[0] << DEL << high[1] << endl;

        fields.clear();
    }
    cout.flush();

    return 0;
}


