#ifndef TASKID_H_
#define TASKID_H_

#include "Task.h"
#include "spatialindex.h"

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
  poly_pair_array_t *poly_pairs ;
  
  // main entry 
  float * crossmatch_cpu();
  float * crossmatch_gpu();
  
  // parsing methods 
  int parse_cpu();
  int parse_polys(poly_array_t *polys, const int did);
  int alloc_poly_array(poly_array_t *polys, const int nr_polys, const int nr_vertices);

  spatial_data_t *load_polys_and_build_index(const int did);
  float *refine_and_do_spatial_op(poly_pair_array_t *poly_pairs, poly_array_t *polys1,poly_array_t *polys2);

 public:
  vector<string>** geom_arr;
  vector<int> nr_vertices;

  JoinTask(int n);

  virtual ~JoinTask();

  bool run(int procType=ExecEngineConstants::CPU, int tid=0);
};

#endif /* TASKID_H_ */
