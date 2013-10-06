#include <stdio.h>
#include <stdlib.h>
#include "justdoit.h"

int main(int argc, char *argv[])
{
	FILE *file1, *file2;
	float *ratios = NULL;
	int i;

	if(argc < 3) {
		printf("USAGE: justdoit filename1 filename2");
		return -1;
	}

	file1 = fopen(argv[1], "r");
	if(!file1) {
		perror("failed to open file1");
		return -1;
	}
	file2 = fopen(argv[2], "r");
	if(!file2) {
		perror("failed to open file2");
		fclose(file1);
		return -1;
	}

	poly_array_t polys1, polys2;
	init_poly_array(&polys1);
	init_poly_array(&polys2);

	if(load_polys(&polys1, file1)) {
		printf("load_polys failed from file1\n");
		exit(1);
	}
	else {
		printf("%d polys loaded from file1\n", polys1.nr_polys);
	}
	if(load_polys(&polys2, file2)) {
		printf("load_polys failed from file2\n");
		exit(1);
	}
	else {
		printf("%d polys loaded from file2\n", polys2.nr_polys);
	}

	poly_pair_array_t poly_pairs;
	init_poly_pair_array(&poly_pairs);
	make_poly_pair_array(&poly_pairs, polys1.nr_polys);

	for(i = 0; i < polys1.nr_polys; i++) {
		poly_pairs.idx1[i] = i;
		poly_pairs.idx2[i] = i;
		mbr_merge(&poly_pairs.mbrs[i], &polys1.mbrs[i], &polys2.mbrs[i]);
	}

	ratios = refine_and_do_spatial_op(&poly_pairs, &polys1, &polys2);
	if(ratios) {
		printf("ratios computed successfully\n");
		for(i = 0; i < polys1.nr_polys; i++) {
			printf("%f\n", ratios[i]);
		}
	}
	else {
		printf("failed to compute ratios\n");
	}

	FREE(ratios);
	fini_poly_pair_array(&poly_pairs);
	fini_poly_array(&polys2);
	fini_poly_array(&polys1);
	fclose(file2);
	fclose(file1);
	return 0;
}
