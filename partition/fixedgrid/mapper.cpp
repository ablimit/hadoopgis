#include "hadoopgis.h"
#include "cmdline.h"


GeometryFactory *gf = NULL;
WKTReader *wkt_reader = NULL;
IStorageManager * storage = NULL;
ISpatialIndex * spidx = NULL;

string doQuery(string polygon) {
    Geometry * poly = wkt_reader->read(polygon);
    const Envelope * env = poly->getEnvelopeInternal();

    low [0] = env->getMinX();
    low [1] = env->getMinY();

    high [0] = env->getMaxX();
    high [1] = env->getMaxY();

    Region r(low, high, 2);

    MyVisitor vis ; 
    spidx->containsWhatQuery(r, vis);

}


vector<Geometry*> genTiles(double min_x, double max_x, double min_y, double  max_y, int x_split, int y_split) {
    vector<Geometry*> tiles;
    stringstream ss;
    double width =  (max_x - min_x) / x_split ;
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
            // cerr << ss.str() << endl;
            tiles.push_back(wkt_reader->read(ss.str()));

            ss.str(string()); // clear the content
        }
    }

    return tiles;
}

vector<string> parsePAIS(string & line) {
    vector<string> vec ;
    size_t pos = line.find_first_of(COMMA,0);
    size_t pos2;
    if (pos == string::npos){
        return vec; // failure
    }

    vec.push_back(line.substr(0,pos)); // tile_id
    pos2=line.find_first_of(COMMA,pos+1);
    vec.push_back(line.substr(pos+1,pos2-pos-1)); // object_id
    pos=pos2;
    vec.push_back(shapebegin + line.substr(pos+2,line.length()- pos - 3) + shapeend);
}

freeObjects() {
    // garbage collection 
    delete wkt_reader ;
    delete gf ; 
    delete spidx;
    delete storage;
}
int main(int argc, char **argv) {
    gengetopt_args_info args_info;
    if (cmdline_parser (argc, argv, &args_info) != 0)
        exit(1) ;

    double min_x = args_info.min_x_arg;
    double max_x = args_info.max_x_arg;
    double min_y = args_info.min_y_arg;
    double max_y = args_info.max_y_arg;
    int x_split = args_info.x_split_arg;
    int y_split = args_info.y_split_arg;
    /* 
       cerr << "min_x "<< min_x << endl; 
       cerr << "max_x "<< max_x << endl; 
       cerr << "min_y "<< min_y << endl; 
       cerr << "max_y "<< max_y << endl; 
       cerr << "x_split "<< x_split << endl; 
       cerr << "y_split "<< y_split << endl; 
       */

    // initlize the GEOS ibjects
    gf = new GeometryFactory(new PrecisionModel(),0);
    wkt_reader= new WKTReader(gf);

    // genrate tile boundaries 
    vector <Geometry*> geom_tiles= genTiles(min_x, max_x, min_y, max_y,x_split,y_split);

    // build spatial index on tile boundaries 
    id_type  indexIdentifier ;
    GEOSDataStream stream(&geom_tiles);
    storage = StorageManager::createNewMemoryStorageManager();
    spidx   = RTree::createAndBulkLoadNewRTree(RTree::BLM_STR, stream, *storage, 
            1.0, // FillFactor
            4,   // IndexCapacity 
            4,   // LeafCapacity
            2, 
            RTree::RV_RSTAR, indexIdentifier);

    // Error checking 
    bool ret = spidx->isIndexValid();
    if (ret == false) 
        std::cerr << "ERROR: Structure is invalid!" << std::endl;


    // process input data 
    string input_line;
    vector<string> fields;
    cerr << "Reading input from stdin..." <<endl;
    string tid ;
    while(cin && getline(cin, input_line) && !cin.eof()){
        fields = parsePAIS(input_line);
        tid = getPID(fields[])

    }


    cout.flush();
    cerr.flush();
    cmdline_parser_free (&args_info); /* release allocated memory */
    return 0; // success
}

RTree::Data* parseInputPolygon(Geometry *p, id_type m_id) {
    double low[2], high[2];
    const Envelope * env = p->getEnvelopeInternal();

    low [0] = env->getMinX();
    low [1] = env->getMinY();

    high [0] = env->getMaxX();
    high [1] = env->getMaxY();

    Region r(low, high, 2);

    //std::cerr << " parseInputPolygon m_id: "  << m_id << std::endl;
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
                //std::cout << "readNextEntry m_id == " << m_id << std::endl;
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
            //std::cout << geometry_collction[d.getIdentifier()] << std::endl;
        }

        void visitData(std::vector<const IData*>& v) {}
        void visitData(std::vector<uint32_t>& v){}
};


