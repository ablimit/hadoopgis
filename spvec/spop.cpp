#include <stdio.h>
#include <stdlib.h>

#include "spop_ispc.h"
#include "spvec.h"
#include "timing.h"

extern double polygon[1016];
extern double ptarray_signed_area(double pa[] , const int npoints);
const int npoints =1015;

//using namespace ispc;

int main(int argc, char** argv) {
        double area[2];
        reset_and_start_timer();
        area[0]= ptarray_signed_area(polygon,npoints);
        double aTime = get_elapsed_mcycles();

        reset_and_start_timer();
        area[1]= ispc::ptarray_signed_area(polygon,npoints);
        double bTime = get_elapsed_mcycles();

        printf("%-40s: [%.2f] M cycles %s, [%.2f] M cycles %s (%.2fx speedup).\n",
               "ST_AREA", aTime, "serial", bTime, "ispc",
               aTime/bTime);
        printf("serial [%.2f], ispc [%.2f]\n", area[0], area[1]);
    
        return 0;
}
