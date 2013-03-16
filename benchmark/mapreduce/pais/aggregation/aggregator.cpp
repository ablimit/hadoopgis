#include "hadoopgis.h"

double avg_area = 0.0;
double avg_perimeter = 0.0;
int size= 0;

/* functions */
void processQuery() { 
    cout << "OSM" << TAB <<avg_area/size<< TAB << avg_perimeter/size<< TAB << size<<endl;
    cout.flush(); 
}

int main(int argc, char **argv) {
    string input_line;
    vector<string> fields;
    double area = 0.0;
    double perimeter = 0.0;
    int c = 0;

    while(cin && getline(cin, input_line) && !cin.eof()){
        boost::split(fields, input_line, boost::is_any_of(TAB));
        area      = boost::lexical_cast< double >( fields[1] ) ;
        perimeter = boost::lexical_cast< double >( fields[2] );
        c = boost::lexical_cast< int >( fields[3] );

        avg_area += area ;
        avg_perimeter += perimeter ;
        size += c ; 
        
        fields.clear();
    }


    processQuery();

    return 0; // success
}

