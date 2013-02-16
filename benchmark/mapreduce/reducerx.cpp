#include <iostream>
#include <vector>
#include <map>

#include "MReducerx.h"


using namespace SpatialIndex;
using namespace std;


//map<string,map<int,vector<string> > > data;

bool readSpatialInput() {
    string input_line;
    int index =0;
    string key ;
    string value;
    polygon shape;
    while(cin && getline(cin, input_line) && !cin.eof()) {
	size_t pos = input_line.find_first_of(tab);
	key = input_line.substr(0,pos);
	index = input_line[pos+1] - offset;   // the very first character denotes join index
	pos = input_line.find_first_of(tab, pos+1);
	value = input_line.substr(pos+2,input_line.size()- pos - 3); // ignore the double quote both at the beginning and at the end.

	//cerr<< "Key: " << key << endl;
	//cerr<< ""Value: " <<value << endl;
	//data[key][index].push_back(value);

	boost::geometry::read_wkt(shapebegin+value+shapeend, shape);
	boost::geometry::correct(shape);
	polydata[key][index].push_back(shape);
    }

    //cerr << "size" << "\t" << polydata.size() << endl;
    return true;
}







int main(int argc, char** argv)
{
    IStorageManager ** storages = NULL ;//new IStorageManager * [data.size()];
    ISpatialIndex ** forest = NULL ;    //new ISpatialIndex * [data.size()];
    id_type * indexIdentifier = NULL;
    polymap::iterator iter;

    bool initilize = true;
    string key ;
    int size = -1;
    bool skip= false;
    int joincar = 0;

    if (! readSpatialInput()) 
    {
	cerr << "Reduce input parsing error...." << endl;
	return 1;
    }

    for (iter= polydata.begin(); iter !=polydata.end(); iter++)
    {
	if (polydata[iter->first].size() > joincar){
	    joincar=polydata[iter->first].size();
	    cerr << "|join|=" << joincar << endl;
	    cerr << "  Note: if this prints more than once then probably there is a bug." << endl;
	}
	//cerr << "size=" <<polydata[iter->first].size() << endl;
    }


    try { 

	// for each tile in the input stream 
	for (iter= polydata.begin(); iter !=polydata.end(); iter++)
	{
	    key  = iter->first;
	    cerr << "[" << key <<"]"<<endl;
	    current_key =key;
	    map<int,vector<polygon> > & images = polydata[key];
	    size = images.size();   // join size

	    if (size< joincar)
	    {
	    cerr << "skipping [" << key <<"] "<<endl;
		continue ;
	    }

	    if (initilize)
	    {
		//we assume all the multiwya joins are the same.
		storages = new IStorageManager * [size];
		forest = new ISpatialIndex * [size];
		indexIdentifier = new id_type [size];
		initilize =false;
	    }

	    // for each spatial data in a set of images 
	    for (int i =0 ;i <size; i++)
	    {
		storages[i]= StorageManager::createNewMemoryStorageManager();
		MRJDataStream stream(&(images[i]), i+1);

		// Create and bulk load a new RTree with dimensionality 2, using memory as the StorageManager and the RSTAR splitting policy.
		forest[i]= RTree::createAndBulkLoadNewRTree( 
			RTree::BLM_STR, stream, *storages[i], FillFactor, IndexCapacity, LeafCapacity, 2, 
			SpatialIndex::RTree::RV_RSTAR, indexIdentifier[i]);

		// std::cerr << *forest[i];
		//std::cerr << "Buffer hits: " << file->getHits() << std::endl;
		//cerr << "Index ID: " << indexIdentifier[i] << endl;

		bool ret = forest[i]->isIndexValid();
		if (ret == false) 
		    cerr << "ERROR: Structure is invalid!" << endl;
		//else 
		//cerr << "The stucture is O.K." << endl;
	    }
	    /*Spatial Join 
	     * Query Type: Clique Query --> One standard baseline is compared against multiple allgorithms.
	     * Then the intersection result is reported individually */
	    MRJVisitor vis;
	    forest[0]->joinMQuery(vis, forest, size);
	    vis.getInfo();
	    // clean up the allocated space
	    for (int i =0 ; i < size; i++)
	    {
		if (NULL != forest[i])
		    delete forest[i];
		if (NULL != storages[i])
		    delete storages[i];
	    }
	}
	/*
	   delete [] forest;
	   delete [] storages;
	   delete [] indexIdentifier;
	   */
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


