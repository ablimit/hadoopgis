#include <stdio.h>
#include <stdlib.h>

#include "spop_ispc.h"
#include "spvec.h"
#include "timing.h"

//double polygon_aos[12] = {3.0,4.0,5.0,11.0,12.0,8.0,9.0,5.0,5.0,6.0,3.0,4.0};
//double polygon_soa[12] = {3.0,5.0,12.0,9.0,5.0,3.0,4.0,11.0,8.0,5.0,6.0,4.0};
extern double ptarray_signed_area_aos(double pa[] , const int npoints);
extern double ptarray_signed_area_soa(double pa[] , const int npoints);
const int npoints =1014;
//const int npoints =10;

//using namespace ispc;

int main(int argc, char** argv) {
  double area[2];
  double aTime = 0.0;
  double bTime = 0.0;
  
  int reps = argc > 1 ? atoi(argv[1]) :100;

  printf("ProgramCount: %d\n",ispc::get_programCount());
  
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
 /* 
  for (int i =0; i<reps; i++) {
    reset_and_start_timer();
    area[1]= ispc::ptarray_signed_area_soa(polygon_soa,npoints);
    bTime += get_elapsed_mcycles();
  }*/
  printf("%-20s: [%.2f] M cycles %s, [%.2f] M cycles %s (%.2fx speedup).\n",
      "ST_AREA", aTime, "serial_soa", bTime, "ispc_soa",
      aTime/bTime);
  printf("%-20s: serial [%.2f], ispc [%.2f]\n", "results", area[0], area[1]);

  return 0;
}
