#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <cmath>
#include <map>
#include <cstdlib> 

// geos
#include <geos/geom/PrecisionModel.h>
#include <geos/geom/GeometryFactory.h>
#include <geos/geom/Geometry.h>
#include <geos/geom/Point.h>
#include <geos/io/WKTReader.h>
#include <geos/io/WKTWriter.h>
#include <geos/opBuffer.h>

#include <spatialindex/SpatialIndex.h>

using namespace std;
using namespace geos;
using namespace geos::io;
using namespace geos::geom;
using namespace geos::operation::buffer; 


// Constants
const int OSM_SRID = 4326;
const int ST_INTERSECTS = 1;
const int ST_TOUCHES = 2;
const int ST_CROSSES = 3;
const int ST_CONTAINS = 4;
const int ST_ADJACENT = 5;
const int ST_DISJOINT = 6;
const int ST_EQUALS = 7;
const int ST_DWITHIN = 8;
const int ST_WITHIN = 9;
const int ST_OVERLAPS = 10;

const int DATABASE_ID_ONE = 1;
const int DATABASE_ID_TWO = 2;

// sepertors for parsing
const string tab = "\t";
const string sep = "\x02"; // ctrl+a

// data type declaration 
map<int, std::vector<Geometry*> > polydata;
map<int, std::vector<string> > rawdata;


double expansion_distance = 0.0;
int JOIN_PREDICATE = 0;
int shape_idx_1 = -1;
int shape_idx_2 = -1;

int join();
void releaseShapeMem();

void tokenize ( const string& str, vector<string>& result,
    const string& delimiters = " ,;:\t", 
    const bool keepBlankFields=false,
    const string& quote="\"\'"
    )
{
  // clear the vector
  if ( false == result.empty() )
  {
    result.clear();
  }

  // you must be kidding
  if (delimiters.empty())
    return ;

  string::size_type pos = 0; // the current position (char) in the string
  char ch = 0; // buffer for the current character
  char delimiter = 0;	// the buffer for the delimiter char which
  // will be added to the tokens if the delimiter
  // is preserved
  char current_quote = 0; // the char of the current open quote
  bool quoted = false; // indicator if there is an open quote
  string token;  // string buffer for the token
  bool token_complete = false; // indicates if the current token is
  // read to be added to the result vector
  string::size_type len = str.length();  // length of the input-string

  // for every char in the input-string
  while ( len > pos )
  {
    // get the character of the string and reset the delimiter buffer
    ch = str.at(pos);
    delimiter = 0;

    bool add_char = true;

    // check ...

    // ... if the delimiter is a quote
    if ( false == quote.empty())
    {
      // if quote chars are provided and the char isn't protected
      if ( string::npos != quote.find_first_of(ch) )
      {
        // if not quoted, set state to open quote and set
        // the quote character
        if ( false == quoted )
        {
          quoted = true;
          current_quote = ch;

          // don't add the quote-char to the token
          add_char = false;
        }
        else // if quote is open already
        {
          // check if it is the matching character to close it
          if ( current_quote == ch )
          {
            // close quote and reset the quote character
            quoted = false;
            current_quote = 0;

            // don't add the quote-char to the token
            add_char = false;
          }
        } // else
      }
    }

    if ( false == delimiters.empty() && false == quoted )
    {
      // if ch is delemiter 
      if ( string::npos != delimiters.find_first_of(ch) )
      {
        token_complete = true;
        // don't add the delimiter to the token
        add_char = false;
      }
    }

    // add the character to the token
    if ( true == add_char )
    {
      // add the current char
      token.push_back( ch );
    }

    // add the token if it is complete
    // if ( true == token_complete && false == token.empty() )
    if ( true == token_complete )
    {
      if (token.empty())
      {
        if (keepBlankFields)
          result.push_back("");
      }
      else 
        // add the token string
        result.push_back( token );

      // clear the contents
      token.clear();

      // build the next token
      token_complete = false;

    }
    // repeat for the next character
    ++pos;
  } // while

  /* 
     cout << "ch: " << (int) ch << endl;
     cout << "token_complete: " << token_complete << endl;
     cout << "token: " << token<< endl;
     */
  // add the final token
  if ( false == token.empty() ) {
    result.push_back( token );
  }
  else if(keepBlankFields && string::npos != delimiters.find_first_of(ch) ){
    result.push_back("");
  }
}

bool readnjoin() 
{
  string input_line;
  string tile_id ;
  string value;
  vector<string> fields;

  int database_id = 0;

  GeometryFactory *gf = new GeometryFactory(new PrecisionModel(),OSM_SRID);
  WKTReader *wkt_reader = new WKTReader(gf);
  Geometry *poly = NULL;
  string previd = "";

  while(cin && getline(cin, input_line) && !cin.eof()) {

    tokenize(input_line, fields,tab,true);
    database_id = atoi(fields[1].c_str());
    tile_id = fields[0];
    // object_id = fields[2];

    // cerr << "fields[0] = " << fields[0] << endl; 
    // cerr << "fields[1] = " << fields[1] << endl; 
    // cerr << "fields[2] = " << fields[2] << endl; 
    // cerr << "fields[9] = " << fields[9] << endl; 
    
    switch(database_id){

      case DATABASE_ID_ONE:
        poly = wkt_reader->read(fields[shape_idx_1]);
        break;

      case DATABASE_ID_TWO:
        poly = wkt_reader->read(fields[shape_idx_2]);
        break;

      default:
        std::cerr << "wrong database id : " << database_id << endl;
        return false;
    }

    /*
    std::stringstream ss;
    for (size_t i =3 ; i < fields.size(); ++i) {
      if (i > 3 ) {
        ss << tab;
      }
      ss << fields[i];
    }
*/
    if (previd.compare(tile_id) !=0 && previd.size() > 0 ) {
     int  pairs = join();
     std::cerr <<  polydata[DATABASE_ID_ONE].size() <<  tab << polydata[DATABASE_ID_TWO].size() <<std::endl;

     std::cerr <<"Tile ID : [" << previd << "] [" << pairs << "]" <<std::endl;
     releaseShapeMem();
     polydata[DATABASE_ID_ONE].clear();
     polydata[DATABASE_ID_TWO].clear();
     rawdata[DATABASE_ID_ONE].clear();
     rawdata[DATABASE_ID_TWO].clear();
    }
      polydata[database_id].push_back(poly);
      rawdata[database_id].push_back(fields[2]);
      previd = tile_id; 

    fields.clear();
  }
    // last tile
     int  pairs = join();
     std::cerr <<  polydata[DATABASE_ID_ONE].size() <<  tab << polydata[DATABASE_ID_TWO].size() <<std::endl;

     std::cerr <<"Tile ID : [" << previd << "] [" << pairs << "]" <<std::endl;
     releaseShapeMem();
     polydata[DATABASE_ID_ONE].clear();
     polydata[DATABASE_ID_TWO].clear();
     rawdata[DATABASE_ID_ONE].clear();
     rawdata[DATABASE_ID_TWO].clear();

  // cerr << "polydata size = " << polydata.size() << endl;
  return true;
}

void releaseShapeMem(){
    int len = polydata[DATABASE_ID_ONE].size();
    for (int i = 0; i < len ; i++) {
      delete polydata[DATABASE_ID_ONE][i];
    }
    len = polydata[DATABASE_ID_TWO].size();
    for (int i = 0; i < len ; i++) {
      delete polydata[DATABASE_ID_TWO][i];
    }
}
bool join_with_predicate(const Geometry * geom1 , const Geometry * geom2, const int jp){
  bool flag = false ; 
  const Envelope * env1 = geom1->getEnvelopeInternal();
  const Envelope * env2 = geom2->getEnvelopeInternal();
  BufferOp * buffer_op1 = NULL ;
  BufferOp * buffer_op2 = NULL ;
  Geometry* geom_buffer1 = NULL;
  Geometry* geom_buffer2 = NULL;

  switch (jp){

    case ST_INTERSECTS:
      flag = env1->intersects(env2) && geom1->intersects(geom2);
      break;

    case ST_TOUCHES:
      flag = geom1->touches(geom2);
      break;

    case ST_CROSSES:
      flag = geom1->crosses(geom2);
      break;

    case ST_CONTAINS:
      flag = env1->contains(env2) && geom1->contains(geom2);
      break;

    case ST_ADJACENT:
      flag = ! geom1->disjoint(geom2);
      break;

    case ST_DISJOINT:
      flag = geom1->disjoint(geom2);
      break;

    case ST_EQUALS:
      flag = env1->equals(env2) && geom1->equals(geom2);
      break;

    case ST_DWITHIN:
      buffer_op1 = new BufferOp(geom1);
      buffer_op2 = new BufferOp(geom2);

      geom_buffer1 = buffer_op1->getResultGeometry(expansion_distance);
      geom_buffer2 = buffer_op2->getResultGeometry(expansion_distance);

      flag = join_with_predicate(geom_buffer1,geom_buffer2, ST_INTERSECTS);
      break;

    case ST_WITHIN:
      flag = geom1->within(geom2);
      break; 

    case ST_OVERLAPS:
      flag = geom1->overlaps(geom2);
      break;

    default:
      std::cerr << "ERROR: unknown spatial predicate " << endl;
      break;
  }
  return flag; 
}

int join() 
{
  // cerr << "---------------------------------------------------" << endl;
  int pairs = 0; 

  // for each tile (key) in the input stream 
  try { 

    std::vector<Geometry*>  & poly_set_one = polydata[DATABASE_ID_ONE];
    std::vector<Geometry*>  & poly_set_two = polydata[DATABASE_ID_TWO];

    int len1 = poly_set_one.size();
    int len2 = poly_set_two.size();

    // cerr << "len1 = " << len1 << endl;
    // cerr << "len2 = " << len2 << endl;

    // should use iterator, update later
    for (int i = 0; i < len1 ; i++) {
      const Geometry* geom1 = poly_set_one[i];

      for (int j = 0; j < len2 ; j++) {
        const Geometry* geom2 = poly_set_two[j];

        // data[key][object_id] = input_line;
        if (join_with_predicate(geom1, geom2, JOIN_PREDICATE))  {
          cout << rawdata[DATABASE_ID_ONE][i] << tab << rawdata[DATABASE_ID_TWO][j] << endl; 
          pairs++;
        }

      } // end of for (int j = 0; j < len2 ; j++) 
    } // end of for (int i = 0; i < len1 ; i++) 	
  } // end of try
  catch (Tools::Exception& e) {
    std::cerr << "******ERROR******" << std::endl;
    std::string s = e.what();
    std::cerr << s << std::endl;
    return -1;
  } // end of catch
  return pairs ;
}

bool cleanup(){ return true; }

bool extractParams(int argc, char** argv ){ 
  /*  if (argc < 4) {
      cerr << "usage: resque [predicate] [shape_idx 1] [shape_idx 2] " <<endl;
      return 1;
      } 

      cerr << "argv[1] = " << argv[1] << endl;
      cerr << "argv[2] = " << argv[2] << endl;
      cerr << "argv[3] = " << argv[3] << endl;
      */
  char *predicate_str = NULL;
  char *distance_str = NULL;
  // get param from environment variables 
  if (argc < 2) {
    if (std::getenv("stpredicate") && std::getenv("shapeidx1") && std::getenv("shapeidx1")) {
      predicate_str = std::getenv("stpredicate");
      shape_idx_1 = strtol(std::getenv("shapeidx1"), NULL, 10) + 2;
      shape_idx_2 = strtol(std::getenv("shapeidx2"), NULL, 10) + 2;
      distance_str = std::getenv("stexpdist");
    } else {
      std::cerr << "ERROR: query parameters are not set in environment variables." << endl;
      return false;
    }
  } 
  // get param from command line arguments
  else if (argc >= 4){
    predicate_str = argv[1];
    if (argc >4)
      predicate_str = argv[4];
    shape_idx_1 = strtol(argv[2], NULL, 10) + 2;
    shape_idx_2 = strtol(argv[3], NULL, 10) + 2;
    // std::cerr << "Params: [" << predicate_str << "] [" << shape_idx_1 << "] " << shape_idx_2 << "]" << std::endl;  
  }
  else {
    return false;
  }

  if (strcmp(predicate_str, "st_intersects") == 0) {
    JOIN_PREDICATE = ST_INTERSECTS;
  } 
  else if (strcmp(predicate_str, "st_touches") == 0) {
    JOIN_PREDICATE = ST_TOUCHES;
  } 
  else if (strcmp(predicate_str, "st_crosses") == 0) {
    JOIN_PREDICATE = ST_CROSSES;
  } 
  else if (strcmp(predicate_str, "st_contains") == 0) {
    JOIN_PREDICATE = ST_CONTAINS;
  } 
  else if (strcmp(predicate_str, "st_adjacent") == 0) {
    JOIN_PREDICATE = ST_ADJACENT;
  } 
  else if (strcmp(predicate_str, "st_disjoint") == 0) {
    JOIN_PREDICATE = ST_DISJOINT;
  }
  else if (strcmp(predicate_str, "st_equals") == 0) {
    JOIN_PREDICATE = ST_EQUALS;
  }
  else if (strcmp(predicate_str, "st_dwithin") == 0) {
    JOIN_PREDICATE = ST_DWITHIN;
    if (NULL != distance_str)
      expansion_distance = atof(distance_str);
    else 
      std::cerr << "ERROR: expansion distance is not set." << std::endl;
    return false;
  }
  else if (strcmp(predicate_str, "st_within") == 0) {
    JOIN_PREDICATE = ST_WITHIN;
  }
  else if (strcmp(predicate_str, "st_overlaps") == 0) {
    JOIN_PREDICATE = ST_OVERLAPS;
  }
  else {
    std::cerr << "unrecognized join predicate " << endl;
    return false ;
  }
  return true;
}

// main body of the engine
int main(int argc, char** argv)
{
  if (!extractParams(argc,argv)) {
    std::cerr <<"ERROR: query parameter extraction error." << std::endl << "Please see documentations, or contact author." << std::endl;
    return 1;
  }

  if (!readnjoin()) {
    std::cerr <<"ERROR: input data parsing error." << std::endl << "Please see documentations, or contact author." << std::endl;
    return 1;
  }
  /*
     int c = join();
     if (c==0) std::cout << std::endl;
     std::cerr <<"Processed pairs: [" <<c << "]" <<std::endl;
     */
  return 0;
}

