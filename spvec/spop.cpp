#include <stdio.h>
#include <stdlib.h>

#include "spop_ispc.h"
#include "spvec.h"
#include "timing.h"

//float polygon_aos[12] = {3.0,4.0,5.0,11.0,12.0,8.0,9.0,5.0,5.0,6.0,3.0,4.0};
//float polygon_soa[12] = {3.0,5.0,12.0,9.0,5.0,3.0,4.0,11.0,8.0,5.0,6.0,4.0};
extern float ptarray_signed_area_aos(float pa[] , const int npoints);
extern float ptarray_signed_area_soa(float pa[] , const int npoints);
const int npoints =1014;
//const int npoints =10;

//using namespace ispc;

int main(int argc, char** argv) {
  float area[2];
  float aTime = 0.0;
  float bTime = 0.0;
  
  int reps = argc > 1 ? atoi(argv[1]) :100;

  printf("ProgramCount [%d], size-of-float[%lu], size-of-float[%lu]\n",ispc::get_programCount(),sizeof(float), sizeof(float));
  
    for (int i =0; i<reps; i++) {
    reset_and_start_timer();
    area[0]= ptarray_signed_area_aos(polygon_aos,npoints);
    aTime += get_elapsed_mcycles();
  }

  for (int i =0; i<reps; i++) {
    reset_and_start_timer();
    area[1]= ispc::ptarray_signed_area_aos(polygon_aos,npoints);
    bTime += get_elapsed_mcycles();
  }

  printf("%-20s: [%.2f] M cycles %s, [%.2f] M cycles %s (%.2fx speedup).\n",
      "ST_AREA", aTime, "serial_aos", bTime, "ispc_aos",
      aTime/bTime);
  printf("%-20s: serial [%.2f], ispc [%.2f]\n", "results", area[0], area[1]);


  aTime = 0.0;
  bTime = 0.0;
  for (int i =0; i<reps; i++) {
    reset_and_start_timer();
    area[0]= ptarray_signed_area_soa(polygon_soa,npoints);
    aTime += get_elapsed_mcycles();
  }
  printf("sequential code is fine.\n");
 
  for (int i =0; i<reps; i++) {
    reset_and_start_timer();
    area[1]= ispc::ptarray_signed_area_soa(polygon_aos,npoints);
    bTime += get_elapsed_mcycles();
  }
  printf("%-20s: [%.2f] M cycles %s, [%.2f] M cycles %s (%.2fx speedup).\n",
      "ST_AREA", aTime, "serial_soa", bTime, "ispc_soa",
      aTime/bTime);
  printf("%-20s: serial [%.2f], ispc [%.2f]\n", "results", area[0], area[1]);

  return 0;
}

