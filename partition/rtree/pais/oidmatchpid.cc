#include <spatialindex/SpatialIndex.h>
#include <boost/algorithm/string/join.hpp>
#include <boost/lexical_cast.hpp>
#include <string>
#include <vector>
#include <map>

using namespace std;
using namespace SpatialIndex;

const unsigned int CHUNCK= 1000000;
const string TAB  = "\t";
const string DASH = "-";
string soid ;

map<id_type,vector<string> > poMap ; // pid --> oid vector

class MyVisitor : public IVisitor
{
public:
	size_t m_indexIO;
	size_t m_leafIO;

public:
	MyVisitor() : m_indexIO(0), m_leafIO(0) {}

	void visitNode(const INode& n)
	{
		if (n.isLeaf()) m_leafIO++;
		else m_indexIO++;
	}

	void visitData(const IData& d)
	{
	    poMap[d.getIdentifier()].push_back(soid);
		//cout << TAB << d.getIdentifier() << DASH << sid;
	}

	void visitData(std::vector<const IData*>& v)
	{
		cout << v[0]->getIdentifier() << " " << v[1]->getIdentifier() << endl;
	}
	void visitData(std::vector<uint32_t>& v)
	{
	    //v.push_back(1);
	    //coll.push_back(v);
	}

	void visitData(string & s)
	{
	    //coll.push_back(s);
	}
};

class MyDataStream : public IDataStream
{
    public:
        MyDataStream(std::string inputFile) : m_pNext(0), m_id(0)
    {
        m_fin.open(inputFile.c_str());

        if (! m_fin)
            throw Tools::IllegalArgumentException("Input file not found.");
        readNextEntry();

    }

        virtual ~MyDataStream()
        {
            if (m_pNext != 0) delete m_pNext;
            m_fin.close();
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
            throw Tools::NotSupportedException("Operation not supported.");
        }

        virtual void rewind()
        {
            if (m_pNext != 0)
            {
                delete m_pNext;
                m_pNext = 0;
            }

            m_fin.seekg(0, std::ios::beg);
            m_id = 0 ;
            readNextEntry();
        }

        void readNextEntry()
        {
            double low[2], high[2];
            double area ; 
            int volume ;


            m_fin >> m_id >> low[0] >> low[1] >> high[0] >> high[1] >> area >> volume;
            //if (! m_fin.good()) ; // skip newlines, etc.


            if (m_fin.good()){
                Region r(low, high, 2);
                m_pNext = new RTree::Data(0, 0 , r, m_id);// store a zero size null poiter.
               // cout << m_id <<TAB << low[0] <<TAB << low[1] <<TAB << high[0] <<TAB << high[1] <<TAB << area <<TAB << volume <<endl;
            }
        }

        std::ifstream m_fin;
        RTree::Data* m_pNext;
        id_type m_id;
        string TAB; 
};

int main(int argc, char** argv)
{
    std::cout.precision(10);
    try
    {
        if (argc != 2)
        {
            std::cerr << "Usage: " << argv[0] << " [partition file]" << std::endl;
            return -1;
        }

        /* 
         * Build spatial index on the partition boundaries */
        int indexCapacity = 4;
        int leafCapacity = 5 ;
        double fillFactor = 0.6;

        IStorageManager* diskfile = StorageManager::createNewMemoryStorageManager();

        MyDataStream stream(argv[1]);
        //cerr << "okay" << endl; 
        cerr.flush();
        // Create and bulk load a new RTree with dimensionality 2, using "file" as
        // the StorageManager and the RSTAR splitting policy.
        id_type indexIdentifier;
        ISpatialIndex* tree = RTree::createAndBulkLoadNewRTree(
                RTree::BLM_STR, stream, *diskfile, fillFactor, indexCapacity, leafCapacity, 2, SpatialIndex::RTree::RV_RSTAR, indexIdentifier);

        //std::cerr << *tree;
        //std::cerr << "Index ID: " << indexIdentifier << std::endl;

        bool ret = tree->isIndexValid();
        if (ret == false) std::cerr << "ERROR: Structure is invalid!" << std::endl;
        //else std::cerr << "The stucture seems O.K." << std::endl;


        /*****************************************************************/
        /*parse the input collection */
	MyVisitor vis;
        double low[2], high[2];
	double temp;
	int oid =-1;
	int sid =-1; 
	unsigned int progress_counter = 0; 
        while(cin && !cin.eof())
        {

            cin >> oid >> sid >> low[0] >> low[1] >> high[0] >> high[1] >> temp;
	    soid.clear();
	    soid = boost::lexical_cast<string>(oid)+ DASH+ boost::lexical_cast<string>(sid);
	    
            Region r = Region(low, high, 2);
            tree->intersectsWithQuery(r, vis);
	    progress_counter++ ; 
	    if (0 ==progress_counter % CHUNCK)
		cerr << TAB <<"Progress Update: " << int (progress_counter / CHUNCK )<< " Million objects passed." <<endl;
	}

	for( map<id_type,vector<string> >::iterator itor = poMap.begin(); itor != poMap.end(); itor++) {
	    cout << itor->first << TAB << boost::algorithm::join(itor->second, TAB) << endl;
	}

	cout.flush();
	/*clean up memory */
	delete tree;
	delete diskfile;

    }
    catch (Tools::Exception& e)
    {
	std::cerr << "******ERROR******" << std::endl;
	std::string s = e.what();
	std::cerr << s << std::endl;
	return -1;
    }

    return 0;
}

