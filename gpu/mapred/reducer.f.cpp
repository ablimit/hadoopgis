#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>

#include <iostream>
#include <fstream>
#include <sstream>

#include "tokenizer.h"
#include "crossmatch.h"
#ifdef GPU
#include "gpu/gpu_spatial.h"
#else
#include "cpu/cpu_spatial.h"
#endif
#include "dbg.h"


int nr_vertices [2] = {0,0};
vector<string>*  geom_meta_array [2] = {NULL,NULL} ;
const string TAB = "\t";
const string COMMA = ",";
const string SPACE = " ";
int gpu_no = 0;

void report(float *ratios, int count)
{ 
  int i;
  for(i = 0; i < count; i++) 
    cout << ratios[i] << endl;
}

void init_spatial_data(spatial_data_t *data)
{
  init_poly_array(&data->polys);
  init_spatial_index(&data->index);
}

void free_spatial_data(spatial_data_t *data)
{
  fini_poly_array(&data->polys);
  fini_spatial_index(&data->index);
  free(data);
}

// Each text line has the following format:
// poly_id, mbr.l mbr.r mbr.b mbr.t, x0 y0, x1 y1, ..., xn yn, x0 y0,
static int parse_polys(poly_array_t *polys, const int did)
{
  if (did >=2)
  {
    cerr << "Error: did >=2" << endl;
    return 1;
  }
  
  vector<string> * poly_meta = geom_meta_array[did];
  int offset = 0;
  vector<string>::size_type i= 0;
	char *buf = new char[10240];
  // read and parse each text line
  for (i =0 ; i< poly_meta->size(); i++ ){
    polys->offsets[i] = offset;
    char *s = buf; 

    // parse record
    //cerr << (*poly_meta)[i] << endl;

    std::size_t len = (*poly_meta)[i].copy(buf,10240);
    buf[len]= ',';
    buf[len+1]= '\0';
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

int load_polys(poly_array_t *polys, const int did)
{
  int retval = 0;
  polys->nr_polys = geom_meta_array[did]->size();
  polys->nr_vertices = nr_vertices[did];

  /* to optimize memory layout on cpu/gpu, we allocate a large continuous space
     that accomodates mbrs, offsets, x and y arrays together; in this manner,
     only one memory movement is needed to transfer all these data from cpu
     to gpu */
  int size_mbrs = polys->nr_polys * sizeof(mbr_t);
  int size_offsets = (polys->nr_polys + 1) * sizeof(int);
  int size_x = polys->nr_vertices * sizeof(int);
  int size_y = polys->nr_vertices * sizeof(int);

  polys->mbrs = (mbr_t*)malloc(size_mbrs + size_offsets + size_x + size_y);
  if(!polys->mbrs) {
    retval = -1;
    return retval;
  }
  polys->offsets = (int *)((char *)(polys->mbrs) + size_mbrs);
  polys->x = (int *)((char *)(polys->offsets) + size_offsets);
  polys->y = (int *)((char *)(polys->x) + size_x);

  // load polys data
  if(parse_polys(polys,did)) {
    retval = -1;
    return retval;
  }
  return retval;
}

spatial_data_t *load_polys_and_build_index(const int did)
{
  spatial_data_t *data = NULL;
  //struct timeval t1, t2;

  data = (spatial_data_t*)malloc(sizeof(spatial_data_t));
  if(!data)
  {
    debug("data is NULL");
    return data;
  }

  init_spatial_data(data);

  poly_array_t *polys = &data->polys;
  spatial_index_t *index = &data->index;
  // load polys || build index
  //gettimeofday(&t1, NULL);
  int en = load_polys(polys,did);
  //gettimeofday(&t2, NULL);
  //cerr << "Time on parsing polygons: " << DIFF_TIME(t1, t2) << " s." <<endl;

  if (en)
  {
    debug("Polygon parsing error %d.",en);
    goto out;
  }
  debug("load_poly set %d -- OK",did);

  //gettimeofday(&t1, NULL);
  en = build_spatial_index(index, polys->mbrs, polys->nr_polys, INDEX_HILBERT_R_TREE);
  //gettimeofday(&t2, NULL);
  //cerr << "Time on building indexes: " << DIFF_TIME(t1, t2) << " s." <<endl;
  if(en)
  {
    debug("indexing error %d", en);
    goto out;
  }
  return data;

out: 
  if(data) {
    free_spatial_data(data);
  }
  return NULL;
}

float *refine_and_do_spatial_op(
    poly_pair_array_t *poly_pairs,
    poly_array_t *polys1,
    poly_array_t *polys2)
{
  cerr << "we are fine in CPU. " <<endl; 
  // we do this operation on gpu/cpu
  return clip(gpu_no,
              poly_pairs->nr_poly_pairs,
              poly_pairs->mbrs,
              poly_pairs->idx1, poly_pairs->idx2,
              polys1->nr_polys + 1, polys1->offsets,
              polys2->nr_polys + 1, polys2->offsets,
              polys1->nr_vertices, polys1->x, polys1->y,
              polys2->nr_vertices, polys2->x, polys2->y);
}

float *crossmatch(float **ratios, int *count)
{
  spatial_data_t *data1 = NULL, *data2 = NULL;
  poly_pair_array_t *poly_pairs = NULL;
  float *result_ratios = NULL;
  struct timeval t1, t2;

  if (geom_meta_array[0]->size()==0 || geom_meta_array[1]->size()==0)
  {
    cerr<< "tile is empty: |T1| = " << geom_meta_array[0]->size() << ", |T2| = " << geom_meta_array[1]->size() <<endl;
    goto out;
  }

  gettimeofday(&t1, NULL);
  // load polys and build indexes
  data1 = load_polys_and_build_index(0);
  if(!data1)
  {
    debug("data1 is NULL.");
    goto out;
  }
  debug("T1 -- OK.");

  data2 = load_polys_and_build_index(1);
  if(!data2)
    goto out;
  debug("T2 -- OK.");

  gettimeofday(&t2, NULL);
  cerr << "Time on loading and building indexes: " << DIFF_TIME(t1, t2) << " s." <<endl;

  gettimeofday(&t1, NULL);
  // filering
  poly_pairs = spatial_filter(&data1->index, &data2->index);
  if(!poly_pairs)
    goto out;
  gettimeofday(&t2, NULL);
  cerr<<"Time on filtering: " <<DIFF_TIME(t1, t2) << " s." <<endl;

  gettimeofday(&t1, NULL);
  // refinement and spatial operations
  result_ratios = refine_and_do_spatial_op(poly_pairs, &data1->polys, &data2->polys);
  if(result_ratios) {
    //		int *temp1 = malloc(poly_pairs->nr_poly_pairs * sizeof(int));
    //		if(!temp1) {
    //			perror("malloc for idx1 failed");
    //			free(result_ratios);
    //			result_ratios = NULL;
    //			goto out;
    //		}
    //		int *temp2 = malloc(poly_pairs->nr_poly_pairs * sizeof(int));
    //		if(!temp2) {
    //			perror("malloc for idx2 failed");
    //			free(temp1);
    //			free(result_ratios);
    //			result_ratios = NULL;
    //			goto out;
    //		}
    //		memcpy(temp1, poly_pairs->idx1, poly_pairs->nr_poly_pairs * sizeof(int));
    //		memcpy(temp2, poly_pairs->idx2, poly_pairs->nr_poly_pairs * sizeof(int));
    //
    //		*idx1 = temp1;
    //		*idx2 = temp2;
    *ratios = result_ratios;
    *count = poly_pairs->nr_poly_pairs;
  }
  gettimeofday(&t2, NULL);
  cerr<< "Time on refinement and spatial op: " <<DIFF_TIME(t1, t2) <<" s." <<endl;

out:
  // free stuff
  if(poly_pairs)
    free_poly_pair_array(poly_pairs);
  if(data1)
    free_spatial_data(data1);
  if(data2)
    free_spatial_data(data2);

  return result_ratios;
}

int main(int argc, char *argv[])
{
  struct timeval t1, t2;
  gettimeofday(&t1, NULL);
  float *ratios = NULL;
  int count = 0;
  string input_line;
  string tid ;
  string prev_tid = "";

  string geom_info;
  geom_meta_array[0] = new vector<string>;
  geom_meta_array[1] = new vector<string>;

  // prepare device
#ifdef GPU
  int nr_gpus = get_gpu_device_count();
  init_device_streams(nr_gpus);
  //printf("Number of CUDA devices: %d\n", nr_gpus);
  if(nr_gpus < 1) {
    cerr << "Error: at least one CUDA device is required to run this program" << endl;
    exit(1);
  }
#endif


  size_t pos, pos2;
  /* vector<float> rat; //collection of all ratios */
  std::ifstream infile("/home/aaji/proj/hadoopgis/gpu/mapred/data/t1372.dat");
  while(getline(infile, input_line)) {
    pos=input_line.find_first_of(TAB,0);
    if (pos == string::npos){
      cerr << "no TAB in the input! We are toasted." << endl;
      return 1; // failure
    }

    tid= input_line.substr(0,pos); // tile id

    // finished reading in a tile data, so perform cross matching
    if (0 != tid.compare(prev_tid) && prev_tid.size()>0) 
    {
      cerr << prev_tid 
          <<": |T1| = " << geom_meta_array[0]->size() <<", |T2| = " << geom_meta_array[1]->size() 
          <<" |V1| = " << nr_vertices[0] <<", |V2| = " << nr_vertices[1] <<endl;
      crossmatch(&ratios,&count);
      if (argc>1)
        report(ratios,count);
      geom_meta_array[0]->clear();
      geom_meta_array[1]->clear();
      nr_vertices[0]= 0;
      nr_vertices[1]= 0;
    }
    // actual geometry info: did,oid,num_ver,mbb, geom
    int i = input_line[pos+1] - '1'; // array position 
    pos2=input_line.find_first_of(COMMA,pos+3); //oid = input_line.substr(pos+3,pos2-pos-3) 
    pos=input_line.find_first_of(COMMA,pos2+1); //num_ver = input_line.substr(pos2+1,pos)
    nr_vertices[i] += std::stoi(input_line.substr(pos2+1,pos-pos2-1));

    geom_info = input_line.substr(pos+1); // nv,mbb,geom
    geom_meta_array[i]->push_back(geom_info);
    prev_tid = tid; 
  }
  // last tile 
  cerr << prev_tid 
      <<": |T1| = " << geom_meta_array[0]->size() <<", |T2| = " << geom_meta_array[1]->size() 
      <<" |V1| = " << nr_vertices[0] <<", |V2| = " << nr_vertices[1] <<endl;
  crossmatch(&ratios,&count);

  if (argc>1)
    report(ratios,count);

  //clear memory 
  for (int i =0; i <2; i++){
    geom_meta_array[i]->clear();
    delete geom_meta_array[i];
  }

  gettimeofday(&t2, NULL);
  cerr<< "Time exec: " <<DIFF_TIME(t1, t2) <<" s." <<endl;
  cout.flush();
  cerr.flush();

  //close device
#ifdef GPU
  fini_device_streams();
#endif

  return 0;
}
