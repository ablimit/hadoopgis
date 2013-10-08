#include <cstring>

// include library header file.
#include <spatialindex/SpatialIndex.h>

using namespace SpatialIndex;
using namespace std;

#define INSERT 1
#define DELETE 0
#define QUERY 2

// example of a Visitor pattern.
// findes the index and leaf IO for answering the query and prints
// the resulting data IDs to stdout.
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

		cout << d.getIdentifier() << endl;
			// the ID of this data entry is an answer to the query. I will just print it to stdout.
	}

	void visitData(std::vector<const IData*>& v)
	{
		cout << v[0]->getIdentifier() << " " << v[1]->getIdentifier() << endl;
	}
};

// example of a Strategy pattern.
// traverses the tree by level.
class MyQueryStrategy : public SpatialIndex::IQueryStrategy
{
private:
	queue<id_type> ids;

public:
	void getNextEntry(const IEntry& entry, id_type& nextEntry, bool& hasNext)
	{
		IShape* ps;
		entry.getShape(&ps);
		Region* pr = dynamic_cast<Region*>(ps);

		delete ps;

		const INode* n = dynamic_cast<const INode*>(&entry);

		// traverse only index nodes at levels 2 and higher.
		if (n != 0 && n->getLevel() > 1)
		{
			for (uint32_t cChild = 0; cChild < n->getChildrenCount(); cChild++)
			{
				ids.push(n->getChildIdentifier(cChild));
			}
		}
		else 
		{
		    if (n !=0 && n->getLevel()==1 )
		    { 
			cout << pr->m_pLow[0] << " " << pr->m_pLow[1] << endl;
			cout << pr->m_pHigh[0] << " " << pr->m_pLow[1] << endl;
			cout << pr->m_pHigh[0] << " " << pr->m_pHigh[1] << endl;
			cout << pr->m_pLow[0] << " " << pr->m_pHigh[1] << endl;
			cout << pr->m_pLow[0] << " " << pr->m_pLow[1] << endl << endl << endl;
			// print node MBRs gnuplot style!
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

// example of a Strategy pattern.
// find the total indexed space managed by the index (the MBR of the root).
class MyQueryStrategy2 : public IQueryStrategy
{
    public:
	Region m_indexedSpace;

    public:
	void getNextEntry(const IEntry& entry, id_type& nextEntry, bool& hasNext)
	{
	    // the first time we are called, entry points to the root.

	    // stop after the root.
	    hasNext = false;

	    IShape* ps;
	    entry.getShape(&ps);
	    ps->getMBR(m_indexedSpace);
	    delete ps;
	}
};

int main(int argc, char** argv)
{
    try
    {
	if (argc != 2)
	{
	    cerr << "Usage: " << argv[0] << "[tree file]" << endl;
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


	cerr << *tree;

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
