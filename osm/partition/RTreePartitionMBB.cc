#include <cstring>

// include library header file.
#include <spatialindex/SpatialIndex.h>

using namespace SpatialIndex;
using namespace std;

// example of a Strategy pattern.
// traverses the tree by level.
class MyQueryStrategy : public SpatialIndex::IQueryStrategy
{
private:
	queue<id_type> ids;

public:
    void getNextEntry(const IEntry& entry, id_type& nextEntry, bool& hasNext)
    {
        const INode* n = dynamic_cast<const INode*>(&entry);

        // traverse only index nodes at levels 0 and higher.
        if (n != NULL) {
            if (n->getLevel() > 0)
            {
                for (uint32_t cChild = 0; cChild < n->getChildrenCount(); cChild++)
                {
                    ids.push(n->getChildIdentifier(cChild));
                }
            }
            else if (n->getLevel() ==0)
            {
                IShape* ps;
                entry.getShape(&ps);
                Region* pr = dynamic_cast<Region*>(ps);
                cout << n->getIdentifier() << " " 
                    << pr->m_pLow[0] << " " << pr->m_pLow[1] << " "
                    << pr->m_pHigh[0] << " " << pr->m_pHigh[1] 
                    << endl;
                // print node MBRs gnuplot style!
                cerr<< "set object rect from " 
                    << pr->m_pLow[0] << "," << pr->m_pLow[1] << " to "<< pr->m_pHigh[0] << "," << pr->m_pHigh[1] << endl;

                delete ps;
            }
            else {
                cerr <<"What the hell is this ? " <<endl;
            }
        }

        if (! ids.empty())
        {
            nextEntry = ids.front(); ids.pop();
            hasNext = true;
        }
        else
        {
            hasNext = false;
        }
    }
};


int main(int argc, char** argv)
{
    try
    {
        if (argc != 2)
        {
            cerr << "Usage: " << argv[0] << " tree_file"<< endl; 
            //query_file query_type [intersection | 10NN | selfjoin]." << endl;
            return -1;
        }

        string baseName = argv[1];
        IStorageManager* diskfile = StorageManager::loadDiskStorageManager(baseName);
        // this will try to locate and open an already existing storage manager.

        StorageManager::IBuffer* file = StorageManager::createNewRandomEvictionsBuffer(*diskfile, 10, false);
        // applies a main memory random buffer on top of the persistent storage manager
        // (LRU buffer, etc can be created the same way).

        // If we need to open an existing tree stored in the storage manager, we only
        // have to specify the index identifier as follows
        ISpatialIndex* tree = RTree::loadRTree(*file, 1);


        MyQueryStrategy qs;
        tree->queryStrategy(qs);

        cerr.flush();
        cout.flush();
        //cerr << *tree;
        //cerr << "Buffer hits: " << file->getHits() << endl;

        delete tree;
        delete file;
        delete diskfile;
        // delete the buffer first, then the storage manager
        // (otherwise the the buffer will fail writting the dirty entries).
    }
    catch (Tools::Exception& e)
    {
        cerr << "******ERROR******" << endl;
        std::string s = e.what();
        cerr << s << endl;
        return -1;
    }
    catch (...)
    {
        cerr << "******ERROR******" << endl;
        cerr << "other exception" << endl;
        return -1;
    }

    return 0;
}

