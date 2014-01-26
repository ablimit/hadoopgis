#include "./SpaceStreamReader.h"

using namespace SpatialIndex;

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

	// IStorageManager* memoryFile = StorageManager::createNewMemoryStorageManager();
	IStorageManager* diskfile = StorageManager::createNewDiskStorageManager(baseName, 4096);
	// Create a new storage manager with the provided base name and a 4K page size.

	StorageManager::IBuffer* file = StorageManager::createNewRandomEvictionsBuffer(*diskfile, 100000, false);
	// applies a main memory random buffer on top of the persistent storage manager
	// (LRU buffer, etc can be created the same way).

	SpaceStreamReader stream(argv[1]);

	// Create and bulk load a new RTree with dimensionality 2, using "file" as
	// the StorageManager and the RSTAR splitting policy.
	id_type indexIdentifier;
	ISpatialIndex* tree = RTree::createAndBulkLoadNewRTree(
		RTree::BLM_STR, stream, *file, fillFactor, indexCapacity, leafCapacity, 2, SpatialIndex::RTree::RV_RSTAR, indexIdentifier);

	//std::cerr << *tree;
	//std::cerr << "Buffer hits: " << file->getHits() << std::endl;
	//std::cerr << "Index ID: " << indexIdentifier << std::endl;

	bool ret = tree->isIndexValid();
	if (ret == false) std::cerr << "ERROR: Structure is invalid!" << std::endl;
	else std::cerr << "The index is O.K." << std::endl;

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

