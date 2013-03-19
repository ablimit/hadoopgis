extern vector<string> geometry_collction ; 
extern vector<string> id_collction ; 
class ContainmentDataStream : public IDataStream
{
    public:
	ContainmentDataStream(vector<string> * invec ) : m_pNext(0), len(0),m_id(0)
    {
	if ( invec->empty())
	    throw Tools::IllegalArgumentException("Input size is ZERO.");
	vec = invec;
	len = vec->size();
	readNextEntry();
    }

	virtual ~ContainmentDataStream()
	{
	    if (m_pNext != 0) delete m_pNext;
	}
	RTree::Data* parseInputPolygon(string strPolygon,id_type m_id) {
	    boost::geometry::read_wkt(strPolygon, nuclei);
	    boost::geometry::envelope(nuclei, mbb);
	    low [0] = boost::geometry::get<boost::geometry::min_corner, 0>(mbb);
	    low [1] = boost::geometry::get<boost::geometry::min_corner, 1>(mbb);

	    high [0] = boost::geometry::get<boost::geometry::max_corner, 0>(mbb);
	    high [1] = boost::geometry::get<boost::geometry::max_corner, 1>(mbb);

	    Region r(low, high, 2);

	    //std::cerr << " parseInputPolygon m_id: "  << m_id << std::endl;
	    return new RTree::Data(0, 0 , r, m_id);// store a zero size null poiter.
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
	    return vec->size();
	    //throw Tools::NotSupportedException("Operation not supported.");
	}

	virtual void rewind()
	{
	    if (m_pNext != 0)
	    {
		delete m_pNext;
		m_pNext = 0;
	    }
	    --m_id;

	    readNextEntry();
	}

	void readNextEntry()
	{
	    if (m_id< len)
	    {
		//std::cout << "readNextEntry m_id == " << m_id << std::endl;
		m_pNext = parseInputPolygon((*vec)[m_id], m_id);
		m_id++;
	    }
	}

	RTree::Data* m_pNext;
	vector<string> * vec; 
	int len;
	id_type m_id;

	double low[2], high[2];
	polygon nuclei;
	box mbb;
};

class MyVisitor : public IVisitor
{
    public:
	int cc =0;
	void getCount() {
	    cout << "Number of Record: " <<cc << endl;
	}
	void visitNode(const INode& n) {}
	void visitData(std::string &s) {}

	void visitData(const IData& d)
	{
	    cc++;
	    //std::cout << id_collction[d.getIdentifier()] << "," <<geometry_collction[d.getIdentifier()] << std::endl;
	}

	void visitData(std::vector<const IData*>& v) {}
	void visitData(std::vector<uint32_t>& v){}
};
