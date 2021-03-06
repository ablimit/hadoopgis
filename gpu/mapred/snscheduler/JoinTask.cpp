#include "JoinTask.h"

JoinTask::JoinTask(int n): N(n) {
  geom_arr = new vector<string>*[N];
  d = new spatial_data_t*[N];
  polys = new poly_array_t*[N];
  indexes = new spatial_index_t*[N];

  for (int i =0 ; i < N ; i++)
  {
    geom_arr[i] = new vector<string>();

    d[i] = new spatial_data_t();
    polys[i] = &(d[i]->polys);
    indexes[i] = &(d[i]->index);
    init_poly_array(polys[i]);
    init_spatial_index(indexes[i]);
    nr_vertices.push_back(0);
  }
  poly_pairs = NULL;
  ratios = NULL;
}

JoinTask::~JoinTask() {

  for (int i =0 ; i < N ; i++) {
    delete geom_arr[i];
    fini_poly_array(polys[i]);
    fini_spatial_index(indexes[i]);
    delete d[i];
  }


  // free stuff
  if(poly_pairs)
    free_poly_pair_array(poly_pairs);
  
  delete [] geom_arr ;
  delete [] d;
  delete [] polys; 
  delete [] indexes;

  nr_vertices.clear();
  if (ratios) free(ratios); 
  std::cerr << "Finished Task: [" <<this->getId() <<"]" <<std::endl;
  std::cerr << "~JoinTask" << std::endl;
}

bool JoinTask::run(int procType, int tid)
{
  if (procType == ExecEngineConstants::CPU) {
    std::cerr << "executing on the CPU engine. " << std::endl;
    this->crossmatch_cpu();
  }
  else if( procType == ExecEngineConstants::GPU ) {
    std::cerr << "executing on the GPU engine. " << std::endl;
    this->crossmatch_gpu();

    //sleep(5/this->getSpeedup(ExecEngineConstants::GPU));
  }
  else 
    std::cerr<< "No idea how to handle. " << std::endl;
  //	std::cout << "Task.id = "<< this->getId() << std::endl;

  if (NULL!=ratios)
    report(ratios, poly_pairs->nr_poly_pairs);
  //this->printDependencies();
  return true;
}


void JoinTask::crossmatch_gpu() {
  // struct timeval t1, t2;
  int ev; 

  if (geom_arr[0]->size()==0 || geom_arr[1]->size()==0)
  {
    std::cerr << "tile is empty: |T1| = " << geom_arr[0]->size() << ", |T2| = " << geom_arr[1]->size() <<std::endl;
    std::cerr.flush();
    return ;
  }

  ev = parse_gpu(); check_debug(ev==0, "Parsing failed.");
  ev = index(); check_debug(ev==0, "Spatial indexing failed.");
  ev = filter();
  ratios = (NULL == poly_pairs) ? NULL : HadoopGIS::GPU::refine(GPUNUMBER, 
                                  poly_pairs->nr_poly_pairs, poly_pairs->mbrs,
                                  poly_pairs->idx1, poly_pairs->idx2,
                                  polys[0]->nr_polys + 1, polys[0]->offsets,
                                  polys[1]->nr_polys + 1, polys[1]->offsets,
                                  polys[0]->nr_vertices, polys[0]->x, polys[0]->y,
                                  polys[1]->nr_vertices, polys[1]->x, polys[1]->y);

}

void JoinTask::crossmatch_cpu()
{
  // struct timeval t1, t2;
  int ev; 

  if (geom_arr[0]->size()==0 || geom_arr[1]->size()==0)
  {
    cerr << "tile is empty: |T1| = " << geom_arr[0]->size() << ", |T2| = " << geom_arr[1]->size() <<endl;
    return;
  }

  ev = parse_cpu(); check_debug(ev==0, "Parsing failed.");
  ev = index(); check_debug(ev==0, "Spatial indexing failed.");
  ev = filter();
  ratios = (NULL == poly_pairs) ? NULL : HadoopGIS::CPU::refine(poly_pairs->nr_poly_pairs,
                                  poly_pairs->mbrs,
                                  poly_pairs->idx1, poly_pairs->idx2,
                                  polys[0]->nr_polys + 1, polys[0]->offsets,
                                  polys[1]->nr_polys + 1, polys[1]->offsets,
                                  polys[0]->nr_vertices, polys[0]->x, polys[0]->y,
                                  polys[1]->nr_vertices, polys[1]->x, polys[1]->y);

}

int JoinTask::alloc_poly_array(poly_array_t *polys, const int nr_polys, const int nr_vertices)
{
  int size_mbrs = nr_polys * sizeof(mbr_t);
  int size_offsets = (nr_polys + 1) * sizeof(int);
  int size_x = nr_vertices * sizeof(int);
  int size_y = nr_vertices * sizeof(int);

  polys->mbrs = (mbr_t *)malloc(size_mbrs + size_offsets + size_x + size_y);
  if(!polys->mbrs) {
    cerr << "failed to allocate memory for poly array" <<endl ;
    return 1;
  }

  polys->nr_polys = nr_polys;
  polys->nr_vertices = nr_vertices;
  polys->offsets = (int *)((char *)(polys->mbrs) + size_mbrs);
  polys->x = (int *)((char *)(polys->offsets) + size_offsets);
  polys->y = (int *)((char *)(polys->x) + size_x);

  return 0;
}

// Each text line has the following format:
// poly_id, mbr.l mbr.r mbr.b mbr.t, x0 y0, x1 y1, ..., xn yn, x0 y0,
int JoinTask::parse_polys(poly_array_t *polys, const int did)
{
  // cerr << "inCPU poly parsing ..." <<endl; 
  const int BUFFER_SIZE = 10240;
  int offset = 0;
  vector<string>::size_type i= 0;
  char *buf = new char[BUFFER_SIZE];
  // read and parse each text line
  for (i =0 ; i< geom_arr[did]->size(); i++ ){
    polys->offsets[i] = offset;
    char *s = buf; 

    // parse record
    //cerr << (*poly_meta)[i] << endl;

    std::size_t len = (*geom_arr[did])[i].copy(buf,BUFFER_SIZE);
    buf[len]= ',';
    buf[len+1]= '\0';

    // skip number of vertices 
    while(*s != ',') s++; s++;

    // parse mbr
    polys->mbrs[i].l = atoi(s);
    while(*s != ' ') s++; s++; 
    polys->mbrs[i].r = atoi(s); 
    while(*s != ' ') s++; s++; 
    polys->mbrs[i].b = atoi(s); 
    while(*s != ' ') s++; s++; 
    polys->mbrs[i].t = atoi(s);
    /* #ifdef DEBUG
     * cerr << "DEBUG Polygon MBR: " << polys->mbrs[i].l << COMMA << polys->mbrs[i].r  
     * << COMMA << polys->mbrs[i].b << COMMA << polys->mbrs[i].t<<endl;
     * #endif
     */

    // parse vertex data
    while (true){
      while(*s != ',') s++; s++;
      if (*s == '\0') break;
      polys->x[offset] = atoi(s);
      while (*s != ' ') s++ ; s++;
      polys->y[offset] = atoi(s);
      offset++;
    }
  }
  // the last offset indexes beyond the end of x,y arrays
  polys->offsets[i] = offset;
  delete [] buf;
  return 0;
}

int JoinTask::parse_cpu()
{
  /* to optimize memory layout on cpu/gpu, we allocate a large continuous space
     that accomodates mbrs, offsets, x and y arrays together; in this manner,
     only one memory movement is needed to transfer all these data from cpu
     to gpu */

  // allocate space for poly data 
  for (int i =0 ; i < N ; i++) 
  {
    int retval_alloc = alloc_poly_array(polys[i], geom_arr[i]->size(), nr_vertices[i]);
    int retval_parse = parse_polys(polys[i], i);

    geom_arr[i]->clear();
    if(retval_alloc || retval_parse) 
      return 1;
  }
  return 0;
}

// spatial indexing methods
int JoinTask::index()
{
  for (int i =0 ; i < N ; i++) 
  {
    int ev = build_spatial_index(indexes[i], polys[i]->mbrs, polys[i]->nr_polys, INDEX_HILBERT_R_TREE);
    if(ev) return 1;
  }
  return 0;
}

int JoinTask::filter()
{
  this->poly_pairs = spatial_filter(&(d[0]->index), &(d[1]->index));
  return 0;
}

int JoinTask::parse_gpu()
{
  for (int i =0 ; i < N ; i++) 
  {
    poly_array_t * p = gpu_parse(GPUNUMBER, nr_vertices[i], geom_arr[i]);
    if (!p) 
      return 1;
    *(polys[i]) = *p ; 
    geom_arr[i]->clear();
  }
  return 0;
}

