#include <spatialindex/SpatialIndex.h>

using namespace SpatialIndex;
using namespace std;

class SpaceStreamReader : public IDataStream
{
  public:
    SpaceStreamReader(std::string inputFile) : m_pNext(0)
  {
    m_fin.open(inputFile.c_str());

    if (! m_fin)
      throw Tools::IllegalArgumentException("Input file not found.");

    readNextEntry();
  }

    virtual ~SpaceStreamReader()
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

      m_fin.seekg(0, std::ios::beg);
      readNextEntry();
    }

    void readNextEntry()
    {
      double low[2], high[2];
      id_type id;


      m_fin >> id >> low[0] >> low[1] >> high[0] >> high[1] ;

      if (m_fin.good())
      {
        Region r(low, high, 2);
        m_pNext = new RTree::Data(0, 0 , r, id);// store a zero size null poiter.
      }
    }

    std::ifstream m_fin;
    RTree::Data* m_pNext;
};

