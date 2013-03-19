#include <numeric>
#include "hadoopgis.h"

map<string,vector<double> > areaMap;
map<string,vector<double> > perimeterMap;


/* functions */
void processQuery() {
    map<string, vector<double> >::iterator itor;
    double avg_area =0.0;
    double avg_perimeter =0.0;
    for(itor= areaMap.begin(); itor != areaMap.end(); itor++) {
	avg_area   = accumulate((itor->second).begin(),(itor->second).end(),0)/(itor->second).size();
	avg_perimeter = accumulate(perimeterMap[itor->first].begin(),perimeterMap[itor->first].end(),0)/(itor->second).size();
	cout << itor->first << TAB <<avg_area<< TAB << avg_perimeter<<endl;
    }
    cout.flush(); 
}

int main(int argc, char **argv) {
    string input_line;
    vector<string> fields;
    
    while(cin && getline(cin, input_line) && !cin.eof()){
	boost::split(fields, input_line, boost::is_any_of(TAB));
	areaMap[fields[0]].push_back(boost::lexical_cast< double >( fields[1] ) );
	perimeterMap[fields[0]].push_back(boost::lexical_cast< double >( fields[2] ));
	fields.clear();
    }

    processQuery();

    return 0; // success
}

