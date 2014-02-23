#include "StdinStreamReader.h"
#include "VecStreamReader.h"
#include <Timer.hpp>
#include<cmath>

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
  if (ac < 2 )
  {
    cerr << "Missing arguments: [bucket_size]" << endl;
    return 1;
  }
  cout.precision(15);
  uint32_t bucket_size = atoi(av[1]);

  int indexCapacity = 20;
  double fillFactor =0.9999;

  vector<Data*> vec;
  StdinStreamReader stream;
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
  // for (unsigned long i = 0 ;i < vec.size();i++)
  //  delete vec[i];

  for (vector<RTree::Data*>::iterator it = tiles.begin() ; it != tiles.end(); ++it) 
  {
    cout << (*it)->m_id << " " << (*it)->m_region <<endl;
    delete *it;
  }
  tiles.clear();
  delete tree;
  delete memoryFile;

  cout.flush();

  return 0; // success
}

