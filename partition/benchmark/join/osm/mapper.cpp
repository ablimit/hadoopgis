#include "hadoopgis.h"


int main(int argc, char **argv) {

    string input_line;
    vector<string> fields;

    while(cin && getline(cin, input_line) && !cin.eof()){

        boost::split(fields, input_line, boost::is_any_of(BAR));
        if (fields[OSM_TILEID].size()> 2 )
            cout << fields[OSM_TILEID]<< TAB << fields[OSM_ID]<< BAR << fields[OSM_OID] << BAR <<fields[OSM_POLYGON]<< endl;
        fields.clear();
    }
    cout.flush();

    return 0; // success
}

