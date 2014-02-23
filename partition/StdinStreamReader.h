#include <spatialindex/SpatialIndex.h>
#include <sstream> 
//#include "tokenizer.h"

using namespace SpatialIndex;
using namespace std;

class StdinStreamReader : public IDataStream
{
  public:
    StdinStreamReader() : m_pNext(0)
  {
    readNextEntry();
  }

    virtual ~StdinStreamReader()
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

      cin.seekg(0, std::ios::beg);
      readNextEntry();
    }

    void readNextEntry()
    {
      double low[2], high[2], center[2];
      id_type id;
      /* if (std::getline(cin, input_line))
         {
         tokenize(input_line, fields, tab, true);
         }
         */
      cin >> input_line >> id >> low[0] >> low[1] >> high[0] >> high[1] ;

      if (cin.good())
      {
        Region r(low, high, 2);
        m_pNext = new RTree::Data(0, 0 , r, id);// store a zero size null poiter.
        // cerr << id << " " << low[0] << " " << low [1]  << " "  << high[0] << " " << high[1] << endl;
      }
    }

    RTree::Data* m_pNext;
    string input_line ;
    /*vector<string> fields;
      stringstream ss;
      string sep = "\t";
      */
};

