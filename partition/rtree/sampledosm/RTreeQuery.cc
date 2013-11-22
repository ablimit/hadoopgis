#include <cstring>

#include <spatialindex/SpatialIndex.h>

using namespace SpatialIndex;
using namespace std;

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
		cout << " " << d.getIdentifier() ;
		// the ID of this data entry is an answer to the query. I will just print it to stdout.
	}

	void visitData(std::vector<const IData*>& v)
	{
		cout << v[0]->getIdentifier() << " " << v[1]->getIdentifier() << endl;
	}
};

int main(int argc, char** argv)
{
	try
	{
		if (argc != 3)
		{
			cerr << "Usage: " << argv[0] << " query_file tree_file" << endl;
			return -1;
		}


		ifstream fin(argv[1]);
		if (! fin)
		{
			cerr << "Cannot open query file " << argv[1] << "." << endl;
			return -1;
		}

		string baseName = argv[2];
		IStorageManager* diskfile = StorageManager::loadDiskStorageManager(baseName);
			// this will try to locate and open an already existing storage manager.

		StorageManager::IBuffer* file = StorageManager::createNewRandomEvictionsBuffer(*diskfile, 10, false);
			// applies a main memory random buffer on top of the persistent storage manager
			// (LRU buffer, etc can be created the same way).

		// If we need to open an existing tree stored in the storage manager, we only
		// have to specify the index identifier as follows
		ISpatialIndex* tree = RTree::loadRTree(*file, 1);

		id_type id;
		double x1, x2, y1, y2;
		double plow[2], phigh[2];

		while (fin)
		{
		    fin >> id >> x1 >> y1 >> x2 >> y2;
		    
		    cerr << "Query Region: (" << x1 << "," << y1 << ") (" << x2 << "," << y2 << ") --- " << id << endl; 
		    
		    if (! fin.good()) continue; // skip newlines, etc.

		    plow[0] = x1; plow[1] = y1;
		    phigh[0] = x2; phigh[1] = y2;

		    MyVisitor vis;

		    Region r = Region(plow, phigh, 2);
		    cout << id ; 
		    tree->intersectsWithQuery(r, vis);
		    cout << endl;
		    // this will find all data that intersect with the query range.

		}


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

