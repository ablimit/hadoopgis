#include <math.h>
#include <dirent.h>
#include <unistd.h>
#include <errno.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <boost/foreach.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/geometry.hpp>
//#include <boost/geometry/geometry.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometries/polygon.hpp>
#include <boost/geometry/geometries/box.hpp>
#include <boost/geometry/domains/gis/io/wkt/wkt.hpp>

#include <spatialindex/SpatialIndex.h>
#include "IndexParam.h"


using namespace SpatialIndex;
using namespace std;
using boost::lexical_cast;
using boost::bad_lexical_cast;

typedef boost::geometry::model::d2::point_xy<int> point;
typedef boost::geometry::model::polygon<point> polygon;
typedef boost::geometry::model::box<point> box;

typedef map<string,map<int,vector<polygon> > > polymap;
typedef map<string,map<int,vector<box> > > boxmap;


const string bar= "|";
const string tab = "\t";
const char comma = ',';

polymap polydata;
box mbb;


vector<string> joinresults;
vector<double> values(4,0.0);

const int DEBUG=0;

string current_key = "" ;

void update_join(double overlap_area, double union_area, double squared_distance) {
    values[0] += overlap_area/union_area;
    values[1] += squared_distance;
    values[2] += overlap_area;
    values[3] += union_area;
}

void reset () {
    values[0] =0.0;
    values[1] =0.0;
    values[2] =0.0;
    values[3] =0.0;
}

string summarize() {
    stringstream oss;
    oss << values[0] << bar<< values[1] << bar << values[2] << bar << values[3];
    reset();
    return oss.str();
}

std::string getDataString(const IData* d) {
    byte* pData = 0;
    uint32_t cLen = 0;
    d->getData(cLen, &pData);
    string s = reinterpret_cast<char*>(pData);
    delete[] pData;
    return s;
}


// example of a Visitor pattern.
class MRJVisitor : public IVisitor
{
    public:
	size_t m_indexIO;
	size_t m_leafIO;
	vector<string> coll;

    private: 
	int m_count;

    public:
	MRJVisitor() : m_indexIO(0), m_leafIO(0) {m_count=0;}

	void visitNode(const INode& n)
	{   
	    if (n.isLeaf()) m_leafIO++;
	    else m_indexIO++;
	}  
	/*
	   void getInfo(){
	   for (int i=0; i< coll.size();i++)
	   {
	   std::cerr << coll[i][0] ;
	   for (int j=1 ;j < coll[i].size() ; j++)
	   {
	   std::cerr << ", " <<coll[i][j] ;
	   }
	   std::cerr << std::endl;
	   }
	   }
	   */

	void getInfo(){

	    double overlap [] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

	    cerr << "[intersecting pairs=" <<coll.size() <<"]" << endl;
	    std::vector<polygon> polyvec;
	    std::vector<polygon> temp;
	    for (int i=0; i< coll.size();i++)
	    {

		std::vector<std::string> strs;
		std::vector<int> indices;

		boost::split(strs, coll[i], boost::is_any_of(","));
		for (int j=0; j< strs.size();j++)
		    indices.push_back(lexical_cast<int>(strs[j]));

		for (int j=1; j<indices.size();j++){ 

		    if (boost::geometry::intersects(polydata[current_key][0][indices[0]],polydata[current_key][j][indices[j]]))
		    {
			boost::geometry::intersection(polydata[current_key][0][indices[0]],polydata[current_key][j][indices[j]],temp);
			BOOST_FOREACH(polygon const& p, temp)
			{
			    overlap[j] += boost::geometry::area(p);
			}
			temp.resize(0);
		    }
		}
	    }
		cout << current_key << tab << "["<<overlap[1]<<bar<<overlap[2]<<bar<<overlap[3]<<bar<<overlap[4]<<bar<<overlap[5]<<"]"<< endl;
	}
	void visitData(const IData& d)
	{   
	    IShape* pS; 
	    d.getShape(&pS);
	    // do something.
	    delete pS; 

	    // data should be an array of characters representing a Region as a string.
	    byte* pData = 0;
	    uint32_t cLen = 0;
	    d.getData(cLen, &pData);
	    // do something.
	    //string s = reinterpret_cast<char*>(pData);
	    //cout << s << endl;
	    delete[] pData;

	    cout << "answer: "<<d.getIdentifier() << endl;
	    // the ID of this data entry is an answer to the query. I will just print it to stdout.
	}

	void visitData(std::vector<const IData*>& v)
	{
	    // to be filled with logic ;
	}

	void visitData(std::vector<uint32_t>& v)
	{
	    v.push_back(1);
	    //coll.push_back(v);
	}

	void visitData(string & s)
	{
	    coll.push_back(s);
	}

};




RTree::Data* parseInputPolygon(polygon &p, id_type m_id) {
    double low[2], high[2];

    boost::geometry::envelope(p, mbb);
    low [0] = boost::geometry::get<boost::geometry::min_corner, 0>(mbb);
    low [1] = boost::geometry::get<boost::geometry::min_corner, 1>(mbb);

    high [0] = boost::geometry::get<boost::geometry::max_corner, 0>(mbb);
    high [1] = boost::geometry::get<boost::geometry::max_corner, 1>(mbb);

    Region r(low, high, 2);

    //std::cerr << " parseInputPolygon m_id: "  << m_id << std::endl;
    return new RTree::Data(0, 0 , r, m_id);// store a zero size null poiter.
}


class MRJDataStream : public IDataStream
{
    public:
	MRJDataStream(vector<polygon> * invec, int tag ) : m_pNext(0), index(0), len(0),m_id(0)
    {
	if ( invec->empty())
	    throw Tools::IllegalArgumentException("Input size is ZERO.");
	vec = invec;
	len = vec->size();
	readNextEntry();
	tagg= tag;
    }

	virtual ~MRJDataStream()
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

	    index =0 ;
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
	vector<polygon> * vec; 
	int len;
	int index ;
	id_type m_id;
	int tagg ;
};

