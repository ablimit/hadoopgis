#ifndef TASKID_H_
#define TASKID_H_

#include "Task.h"
#include "cpu_refine.h"
extern "C" {
#include "spatialindex.h"
}

extern "C" void init_poly_array(poly_array_t *polys);

// spatial data: polygon array and its spatial index
typedef struct spatial_data_struct
{
	poly_array_t polys;
	spatial_index_t index;
} spatial_data_t;


class JoinTask: public Task {
 private:
  int N ;
  int res_size;
  spatial_data_t **d;
  poly_array_t **polys; // sudo pointers to the spatial_data_t 
  spatial_index_t **indexes; 
  poly_pair_array_t *poly_pairs;
  
  // main entry 
  float * crossmatch_cpu();
  float * crossmatch_gpu();
  
  // parsing methods 
  int parse_cpu();
  int parse_polys(poly_array_t *polys, const int did);
  int alloc_poly_array(poly_array_t *polys, const int nr_polys, const int nr_vertices);
  
  // spatial indexing methods
  int index();

  //filter 
  int filter();
  


 public:
  vector<string>** geom_arr;
  vector<int> nr_vertices;

  JoinTask(int n);

  virtual ~JoinTask();

  bool run(int procType=ExecEngineConstants::CPU, int tid=0);

  static void report(float* ratios, int count )
  { 
    int i;
    for(i = 0; i < count; i++) 
      cout << ratios[i] << endl;
  }
};

#endif /* TASKID_H_ */
