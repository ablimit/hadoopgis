#include "hadoopgis.h"
#include "cmdline_reducer.h"


GeometryFactory *gf = NULL;
WKTReader *wkt_reader = NULL;
IStorageManager * storage = NULL;
ISpatialIndex * spidx = NULL;
map<int,string> id_polygon ;
vector<int> hits ; 


RTree::Data* parseInputPolygon(Geometry *p, id_type m_id) {
    double low[2], high[2];
    const Envelope * env = p->getEnvelopeInternal();

    low [0] = env->getMinX();
    low [1] = env->getMinY();

    high [0] = env->getMaxX();
    high [1] = env->getMaxY();

    Region r(low, high, 2);

    //std::cerr << "parseInputPolygon m_id: "  << m_id << std::endl;
    return new RTree::Data(0, 0 , r, m_id);// store a zero size null poiter.
}


class GEOSDataStream : public IDataStream
{
    public:
        GEOSDataStream(vector<Geometry*> * invec) : m_pNext(0), index(0), len(0),m_id(0)
    {
        if ( invec->empty())
            throw Tools::IllegalArgumentException("Input size is ZERO.");
        vec = invec;
        len = vec->size();
        readNextEntry();
    }
        virtual ~GEOSDataStream()
        {
            if (m_pNext != 0) delete m_pNext;
        }

        virtual IData* getNext()
        {
            if (m_pNext == 0) return 0;

            RTree::Data* ret = m_pNext;
            m_pNext = 0;
            readNextEntry();
            return ret;
        }

        virtual bool hasNext()
        {
            return (m_pNext != 0);
        }

        virtual uint32_t size()
        {
            return vec->size();
            //throw Tools::NotSupportedException("Operation not supported.");
        }

        virtual void rewind()
        {
            if (m_pNext != 0)
            {
                delete m_pNext;
                m_pNext = 0;
            }

            index = 0;
            m_id  = 0;
            readNextEntry();
        }

        void readNextEntry()
        {
            if (index < len)
            {
                //std::cerr<< "readNextEntry m_id == " << m_id << std::endl;
                m_pNext = parseInputPolygon((*vec)[index], m_id);
                index++;
                m_id++;
            }
        }

        RTree::Data* m_pNext;
        vector<Geometry*> * vec; 
        int len;
        int index ;
        id_type m_id;
};


class MyVisitor : public IVisitor
{
    public:
        void visitNode(const INode& n) {}
        void visitData(std::string &s) {}

        void visitData(const IData& d)
        {
            hits.push_back(d.getIdentifier());
            //std::cout << d.getIdentifier()<< std::endl;
        }

        void visitData(std::vector<const IData*>& v) {}
        void visitData(std::vector<uint32_t>& v){}
};



void doQuery(Geometry* poly) {
    double low[2], high[2];
    const Envelope * env = poly->getEnvelopeInternal();

    low [0] = env->getMinX();
    low [1] = env->getMinY();

    high [0] = env->getMaxX();
    high [1] = env->getMaxY();

    Region r(low, high, 2);

    // clear the result container 
    hits.clear();

    MyVisitor vis ; 
    //spidx->containsWhatQuery(r, vis);
    spidx->intersectsWithQuery(r, vis);
}


vector<Geometry*> genTiles(double min_x, double max_x, double min_y, double  max_y, int x_split, int y_split) {
    vector<Geometry*> tiles;
    stringstream ss;
    double width =  (max_x - min_x) / x_split;
    //cerr << "Tile width" << SPACE <<width <<endl;
    double height = (max_y - min_y)/y_split ;
    //cerr << "Tile height" << SPACE << height <<endl;

    for (int i =0 ; i< x_split ; i++)
    {
        for (int j =0 ; j< y_split ; j++)
        {
            // construct a WKT polygon 
            ss << shapebegin ;
            ss << min_x + i * width ;     ss << SPACE ; ss << min_y + j * height;     ss << COMMA;
            ss << min_x + i * width ;     ss << SPACE ; ss << min_y + (j+1) * height; ss << COMMA;
            ss << min_x + (i+1) * width ; ss << SPACE ; ss << min_y + (j+1) * height; ss << COMMA;
            ss << min_x + (i+1) * width ; ss << SPACE ; ss << min_y + j * height;     ss << COMMA;
            ss << min_x + i * width ;     ss << SPACE ; ss << min_y + j * height;
            ss << shapeend ;
            //cerr << ss.str() << endl;
            tiles.push_back(wkt_reader->read(ss.str()));
            ss.str(string()); // clear the content
        }
    }

    return tiles;
}

vector<string> parsePAIS(string & line, string delim) {
    vector<string> vec ;
    boost::split(vec,line, boost::is_any_of(delim));
    return vec ;
}

void freeObjects() {
    // garbage collection 
    delete spidx;
    delete storage;
}

void emitHits(Geometry* poly) {
    double low[2], high[2];
    const Envelope * env = poly->getEnvelopeInternal();

    low [0] = env->getMinX();
    low [1] = env->getMinY();

    high [0] = env->getMaxX();
    high [1] = env->getMaxY();

    stringstream ss; // tile_id ; 
    ss << low[0] ;
    ss << DASH ;
    ss << low[1] ;
    ss << DASH ;
    ss << high[0] ;
    ss << DASH ;
    ss<< high[1] ;

    cerr << "Hit size: " << hits.size() <<endl ;

    for (int i = 0 ; i < hits.size(); i++ ) 
    {
        cout << ss.str() << TAB << hits[i]  << TAB << id_polygon[hits[i]] << TAB << endl ;
    }
}

void extractSpatialUniverse(string tid, double & min_x, double & max_x, double & min_y, double &max_y){
    vector<string> strs ; 
    boost::split(strs,tid, boost::is_any_of(DASH));
    min_x  = boost::lexical_cast< double >( strs[0] ); 
    min_y  = boost::lexical_cast< double >( strs[1] ); 
    max_x  = boost::lexical_cast< double >( strs[2] ); 
    max_y  = boost::lexical_cast< double >( strs[3] ); 
    //return true ;
}


bool buildIndex(vector<Geometry*> & geom_polygons) {
    // build spatial index on tile boundaries 
    id_type  indexIdentifier;
    GEOSDataStream stream(&geom_polygons);
    storage = StorageManager::createNewMemoryStorageManager();
    spidx   = RTree::createAndBulkLoadNewRTree(RTree::BLM_STR, stream, *storage, 
	    FillFactor,
	    IndexCapacity,
	    LeafCapacity,
	    2, 
	    RTree::RV_RSTAR, indexIdentifier);

    // Error checking 
    return spidx->isIndexValid();
}


void process(vector<Geometry*> & geom_polygons, string & tid, int x_split, int y_split) {
    double min_x, max_x, min_y, max_y ;
    // build spatial index for input polygons
    cerr << "Number of objects: " << geom_polygons.size() << endl;

    bool ret = buildIndex(geom_polygons);
    if (ret == false ) 
	std::cerr << "ERROR: Structure is invalid!" << std::endl;
    //else 
	//cerr << "GRIDIndex Generated successfully." << endl;

    // genrate tile boundaries 
    extractSpatialUniverse(tid,min_x,max_x,min_y,max_y);
    vector <Geometry*> geom_tiles= genTiles(min_x, max_x, min_y, max_y,x_split,y_split);
    cerr << "Number of tiles: " << geom_tiles.size() << endl;

    // retrive objects which are intersects with this tile 
    for(std::vector<Geometry*>::iterator it = geom_tiles.begin(); it != geom_tiles.end(); ++it) {
	doQuery(*it);
	emitHits(*it);
    }

}

int main(int argc, char **argv) {
    gengetopt_args_info args_info;
    if (cmdline_parser (argc, argv, &args_info) != 0)
	exit(1) ;

    double min_x, max_x, min_y, max_y ;
    int x_split = args_info.x_split_arg;
    int y_split = args_info.y_split_arg;


    // initlize the GEOS ibjects
    gf = new GeometryFactory(new PrecisionModel(),0);
    wkt_reader= new WKTReader(gf);

    // process input data 
    vector <Geometry*> geom_polygons;
    string input_line;
    vector<string> fields;
    cerr << "Reading input from stdin..." <<endl; 
    int i =0; 
    string tid = "";

    while(cin && getline(cin, input_line) && !cin.eof()){
	fields = parsePAIS(input_line,TAB);

	if (fields[0] != tid && tid.length() > 0 ){

	    process(geom_polygons, tid, x_split, y_split);
	    
	    // reset allocated resources 
	    freeObjects();
	    geom_polygons.clear();
	    id_polygon.clear();
	    i =0; 

	}

	geom_polygons.push_back(wkt_reader->read(fields[2]));
	id_polygon[i] = fields[2];
	tid = fields[0];
	i++;

    }
    
    // last tile 
    process(geom_polygons, tid, x_split, y_split);
    freeObjects();

    cout.flush();
    cerr.flush();
    cmdline_parser_free (&args_info); /* release allocated memory */
    delete wkt_reader ;
    delete gf ; 
    return 0; // success
}

