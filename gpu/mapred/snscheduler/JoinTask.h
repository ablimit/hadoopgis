#ifndef TASKID_H_
#define TASKID_H_

#define GPUNUMBER 0
#include "Task.h"
#include <assert.h>

// cpu related headers 
#include "cpu_refine.h"
extern "C" {
#include "spatialindex.h"
}
// extern "C" void init_poly_array(poly_array_t *polys);
// extern poly_array_t *gpu_parse(int dno, const int nv, std::vector<std::string> * poly_vec);

// GPU related headers
#include "gpu_refine.h"
#include "parser.cuh"
#include "util.cuh"

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
  float * ratios ;
  
  // main entry 
  void crossmatch_cpu();
  void crossmatch_gpu();
  
  // parsing methods 
  int parse_cpu();
  int parse_gpu();
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
    if (ratios != NULL && count > 0)
    {
    std::cerr << "number of intersection objects: " << count << std::endl;
    int i;
    for(i = 0; i < count; i++) 
      cout << ratios[i] << endl;
    }
  }
};

#endif /* TASKID_H_ */
