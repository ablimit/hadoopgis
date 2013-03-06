#include <spatialindex/SpatialIndex.h>

using namespace std;
using namespace SpatialIndex;

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

            
            m_fin >> m_id >> low[0] >> low[1] >> high[0] >> high[1];
            
            if (m_fin.good())
            {
                //m_id++;
                Region r(low, high, 2);
                m_pNext = new RTree::Data(0, 0 , r, m_id);// store a zero size null poiter.
            }
        }

        std::ifstream m_fin;
        RTree::Data* m_pNext;
        id_type m_id;
};

int main(int argc, char** argv)
{
    try
    {
        if (argc != 6)
        {
            std::cerr << "Usage: " << argv[0] << " input_file tree_file indexCapacity leafCapacity fillFactor." << std::endl;
            return -1;
        }

        std::string baseName = argv[2];
        int indexCapacity = atoi(argv[3]);
        int leafCapacity = atoi(argv[4]);
        double fillFactor = atof(argv[5]);

        IStorageManager* diskfile = StorageManager::createNewDiskStorageManager(baseName, 4096);
        // Create a new storage manager with the provided base name and a 4K page size.

        StorageManager::IBuffer* file = StorageManager::createNewRandomEvictionsBuffer(*diskfile, 10, false);
        // applies a main memory random buffer on top of the persistent storage manager
        // (LRU buffer, etc can be created the same way).

        MyDataStream stream(argv[1]);
        cerr << "okay" << endl; 
        cerr.flush();
        // Create and bulk load a new RTree with dimensionality 2, using "file" as
        // the StorageManager and the RSTAR splitting policy.
        id_type indexIdentifier;
        ISpatialIndex* tree = RTree::createAndBulkLoadNewRTree(
                RTree::BLM_STR, stream, *file, fillFactor, indexCapacity, leafCapacity, 2, SpatialIndex::RTree::RV_RSTAR, indexIdentifier);

        std::cerr << *tree;
        std::cerr << "Buffer hits: " << file->getHits() << std::endl;
        std::cerr << "Index ID: " << indexIdentifier << std::endl;

        bool ret = tree->isIndexValid();
        if (ret == false) std::cerr << "ERROR: Structure is invalid!" << std::endl;
        else std::cerr << "The stucture seems O.K." << std::endl;

        delete tree;
        delete file;
        delete diskfile;
        // delete the buffer first, then the storage manager
        // (otherwise the the buffer will fail trying to write the dirty entries).
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

