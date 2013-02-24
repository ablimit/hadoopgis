#include "hadoopgis.h"
#include "vecstream.h"

#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>

const string paisUID = "gbm1.1";
const string tileID = "gbm1.1-0000040960-0000040960";
const string region ="POLYGON((40960 40960, 41984 40960,  41984 41984, 40960 41984, 40960 40960))" ;
vector<string> geometry_collction ; 
double plow[2], phigh[2];

int main(int argc, char **argv) {
    vector<string> strs;
    boost::split(strs, tileID, boost::is_any_of("-"));
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

    polygon poly; 
    boost::geometry::read_wkt(ss.str(), poly);

    


    try
    {
    cout << boost::lexical_cast< double >( strs[1] ) <<endl;
    cout << boost::lexical_cast< double >( strs[2] ) <<endl;
    }
    catch( const boost::bad_lexical_cast & )
    {
	cerr<< "Error: can't cast !" <<endl; 
    }

}

