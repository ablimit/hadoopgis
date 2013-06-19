#include <spatialindex/SpatialIndex.h>

using namespace std;
using namespace SpatialIndex;

const string TAB  = "\t";
const string DASH = "-";
int sid = 0 ; 

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
        /*
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
        */

		cout << TAB << d.getIdentifier() << DASH << sid<<endl;
			// the ID of this data entry is an answer to the query. I will just print it to stdout.
	}

	void visitData(std::vector<const IData*>& v)
	{
		cout << v[0]->getIdentifier() << " " << v[1]->getIdentifier() << endl;
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
                cout << m_id <<TAB << low[0] <<TAB << low[1] <<TAB << high[0] <<TAB << high[1] <<TAB << area <<TAB << volume <<endl;
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
        cerr << "okay" << endl; 
        cerr.flush();
        // Create and bulk load a new RTree with dimensionality 2, using "file" as
        // the StorageManager and the RSTAR splitting policy.
        id_type indexIdentifier;
        ISpatialIndex* tree = RTree::createAndBulkLoadNewRTree(
                RTree::BLM_STR, stream, *diskfile, fillFactor, indexCapacity, leafCapacity, 2, SpatialIndex::RTree::RV_RSTAR, indexIdentifier);

        std::cerr << *tree;
        std::cerr << "Index ID: " << indexIdentifier << std::endl;

        bool ret = tree->isIndexValid();
        if (ret == false) std::cerr << "ERROR: Structure is invalid!" << std::endl;
        else std::cerr << "The stucture seems O.K." << std::endl;


        /*****************************************************************/
        /*parse the input collection */
		MyVisitor vis;
        double low[2], high[2];
        int oid; 
        
        while(cin && !cin.eof())
        {
            cin >> oid >> sid >> low[0] >> low[1] >> high[0] >> high[1];
            Region r = Region(low, phigh, 2);
            tree->intersectsWithQuery(r, vis);
            cout << endl;
        }

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

