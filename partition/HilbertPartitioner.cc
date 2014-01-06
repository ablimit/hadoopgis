#include "./SpaceStreamReader.h"

using namespace SpatialIndex::RTree;

std::vector<Region*> buffer;

void freeObjects()
{
  // free obbjects 
  for(std::vector<Region*>::iterator it = buffer.begin(); it != buffer.end(); ++it)
  {
    delete *it;
  }
}

// function implementations 
Region calculatePartition(){
  Region r ;
  
  if (buffer.size()>0){
    r = *(buffer[0]);
    //for ( std::vector<Region*>::iterator iter = buffer.begin(); iter != buffer.end(); ++iter )
    for ( size_t i =1 ; i< buffer.size(); i++)
      r.combineRegion(*(buffer[i]));
  }
  return r;
}

// main method

int main(int argc, char** argv)
{
  if (argc < 3)
  {
    std::cerr << "Usage: " << argv[0] << " [input_file] [partition_size]" << std::endl;
    return 1;
  }
  const uint32_t bucket_size= strtoul (argv[2], NULL, 0);
  std::cerr << "bucket_size = " <<  bucket_size  << std::endl;
  uint64_t recs = 0 ; 
  uint32_t pid = 0;

  SpaceStreamReader stream(argv[1]);

  while (stream.hasNext())
  {
    Data* d = reinterpret_cast<Data*>(stream.getNext());
    if (d == 0)
      throw Tools::IllegalArgumentException(
          "bulkLoadUsingRPLUS: RTree bulk load expects SpatialIndex::RTree::Data entries."
          );
    Region *obj = d->m_region.clone();
    buffer.push_back(obj);
    delete d;

    if ((++recs % bucket_size) == 0){
      cout << ++pid << " " << calculatePartition()<< endl;
      freeObjects();
      buffer.clear();
    }

    if (recs % 5000000 == 0)
    std::cerr << "Processed records " <<  recs  << std::endl;
  }

  if (buffer.size()>0){
  cout << ++pid << " " << calculatePartition() << endl;
  freeObjects();
  buffer.clear();
  }
  return 0;
}

