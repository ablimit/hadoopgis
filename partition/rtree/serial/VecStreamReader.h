#include <spatialindex/SpatialIndex.h>

using namespace SpatialIndex::RTree;

class VecStreamReader : public IDataStream
{
  public:
    VecStreamReader(vector<Data*> * coll) : m_pNext(0)
  {
    if (coll->size()==0)
      throw Tools::IllegalArgumentException("Input file not found.");

    sobjects = coll;
    index = 0 ;
    readNextEntry();
  }

    virtual ~VecStreamReader()
    {
      if (m_pNext != 0) delete m_pNext;
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
      index = 0;
      readNextEntry();
    }

    void readNextEntry()
    {
      if (index < sobjects->size())
      {
        m_pNext = (*sobjects)[index];
        index++;
      }
    }

    vector<Data*> * sobjects;
    vector<Data*>::size_type index ;
    RTree::Data* m_pNext;
};

