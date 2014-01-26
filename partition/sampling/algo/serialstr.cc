#include "SpaceStreamReader.h"
#include "VecStreamReader.h"
#include <Timer.hpp>
#include<cmath>
#include <boost/program_options.hpp>


// using namespace boost;
namespace po = boost::program_options;


vector<Data*> tiles;
id_type bid = 1; //bucket id


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
        tiles.push_back(new Data(0, 0 , *pr, bid++));
        /*
         * cout << n->getIdentifier() << " " 
         << pr->m_pLow[0] << " " << pr->m_pLow[1] << " "
         << pr->m_pHigh[0] << " " << pr->m_pHigh[1] 
         << endl;
        // print node MBRs gnuplot style!

         * cerr<< "set object rect from " 
         << pr->m_pLow[0] << "," << pr->m_pLow[1] << " to "<< pr->m_pHigh[0] << "," << pr->m_pHigh[1] << endl;
         */
        delete ps;
      }
      /*
         else {
         cerr <<"What the hell is this ? " <<endl;
         }*/
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



int main(int ac, char* av[]){
  uint32_t bucket_size ;
  int indexCapacity = 20;
  double fillFactor =0.9999;
  string inputPath;

  try {
    po::options_description desc("Options");
    desc.add_options()
        ("help", "this help message")
        ("bucket,b", po::value<uint32_t>(&bucket_size), "Expected bucket size")
        ("input,i", po::value<string>(&inputPath), "Data input file path")
        // ("index", po::value<string>(&indexPath), "index file path")
        ;

    po::variables_map vm;        
    po::store(po::parse_command_line(ac, av, desc), vm);
    po::notify(vm);    

    if ( vm.count("help") || (! vm.count("bucket")) || (!vm.count("input")) ) {
      cerr << desc << endl;
      return 0; 
    }

    cerr << "Bucket size: "<<bucket_size <<endl;
    cerr << "Data: "<<inputPath <<endl;
    //return 0; 

  }
  catch(exception& e) {
    cerr << "error: " << e.what() << "\n";
    return 1;
  }
  catch(...) {
    cerr << "Exception of unknown type!\n";
    return 1;
  }
  vector<Data*> vec;
  SpaceStreamReader stream(inputPath);
  while (stream.hasNext())
  {
    Data* d = reinterpret_cast<Data*>(stream.getNext());
    if (d == 0)
      throw Tools::IllegalArgumentException(
          "bulkLoadUsingRPLUS: RTree bulk load expects SpatialIndex::RTree::Data entries."
          );
    vec.push_back(d);
  }

  Timer t;                         
  // build in memory Tree
  VecStreamReader vecstream(&vec);
  IStorageManager* memoryFile = StorageManager::createNewMemoryStorageManager();
  id_type indexIdentifier;

  ISpatialIndex* tree = RTree::createAndBulkLoadNewRTree(
      RTree::BLM_STR, vecstream, *memoryFile, fillFactor, indexCapacity, bucket_size, 2, SpatialIndex::RTree::RV_RSTAR, indexIdentifier);

  MyQueryStrategy qs;
  tree->queryStrategy(qs);
  double elapsed_time = t.elapsed();
  cerr << "stat:ptime," << bucket_size << "," << tiles.size() <<"," << elapsed_time << endl;

  // cleanup allocated memory
  //for (size_type i =0 ;i < vec.size();i++)
  //  delete vec[i];

  for (vector<RTree::Data*>::iterator it = tiles.begin() ; it != tiles.end(); ++it) 
  { 
    cout << (*it)->m_id << " " << (*it)->m_region << endl;
    delete *it;
  }

  delete tree;
  delete memoryFile;

  cout.flush();

  return 0; // success
}

