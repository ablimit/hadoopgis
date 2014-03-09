#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>

#include <string>
#include <vector>
#include <iostream>

#include "crossmatch.h"
#include "cuda/cuda_spatial.h"

using namespace std;

int nr_polys, nr_vertices;
vector<string>*  geom_meta_array [2] = {NULL,NULL} ;

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
// Note: there is a tailing `,' at the end of each line, to ease parsing
static int parse_polys(poly_array_t *polys, const int did)
{
  if (did >=2)
  {
    cerr << "Error: did >=2" << endl;
    return 1;
  }

  vector<string> * poly_meta = geom_meta_array[did];
	int offset = 0;
  size_t i= 0;
  int nv = 0;

	// read and parse each text line
  for (i =0 ; i< poly_meta->size(); i++ ){
	  char *p = (*poly_meta)[i].c_str();
    polys->offsets[i] = offset;
    
    // read in number of vertices
		sscanf(p, "%d", &nv);
		
    // skip nvtext
		while(*p != ',') p++;
		p++;

		// read in mbr data
		sscanf(p, "%d %d %d %d", &polys->mbrs[i].l, &polys->mbrs[i].r,
			&polys->mbrs[i].b, &polys->mbrs[i].t);

		// skip mbr text
		while(*p != ',') p++;
		p++;

		// parse vertex data
    for (int j =0; j < nv; j++)
		{
			sscanf(p, "%d %d", &polys->x[offset], &polys->y[offset]);
			offset++;
      // skip the already processed vertices text 
			while(*p!= ',') p++;
      p++;
		} //while(*p != '\n' && *p != '\0');
	}

	// the last offset indexes beyond the end of x,y arrays
	polys->offsets[i] = offset;
	return 0;
}

int load_polys(poly_array_t *polys, const int did)
{
	int retval = 0;
  polys->nr_polys = nr_polys;
  polys->nr_vertices = nr_vertices;

	/* to optimize memory layout on cpu/gpu, we allocate a large continuous space
	that accomodates mbrs, offsets, x and y arrays together; in this manner,
	only one memory movement is needed to transfer all these data from cpu
	to gpu */
	int size_mbrs = polys->nr_polys * sizeof(mbr_t);
	int size_offsets = (polys->nr_polys + 1) * sizeof(int);
	int size_x = polys->nr_vertices * sizeof(int);
	int size_y = polys->nr_vertices * sizeof(int);

	polys->mbrs = malloc(size_mbrs + size_offsets + size_x + size_y);
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

	data = malloc(sizeof(spatial_data_t));
	if(!data)
		goto error;
	init_spatial_data(data);

	// load polys
	poly_array_t *polys = &data->polys;
	if(load_polys(polys,did))
		goto error;

	// build index
	spatial_index_t *index = &data->index;
	if(build_spatial_index(index, polys->mbrs, polys->nr_polys, INDEX_R_TREE))
		goto error;

	// bingo!
	goto success;

error:
	if(data) {
		free_spatial_data(data);
		data = NULL;
	}

success:
	return data;
}

float *refine_and_do_spatial_op(
	poly_pair_array_t *poly_pairs,
	poly_array_t *polys1,
	poly_array_t *polys2)
{
	// we do this operation on gpu
	return cuda_clip(
		poly_pairs->nr_poly_pairs, poly_pairs->mbrs,
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
	int nr_pairs = 0;
	struct timeval t1, t2;

	gettimeofday(&t1, NULL);
	// load polys and build indexes
	data1 = load_polys_and_build_index(0);
	if(!data1)
		goto out;
	data2 = load_polys_and_build_index(1);
	if(!data2)
		goto out;
	gettimeofday(&t2, NULL);
	printf("Time on loading and building indexes: %lf s\n", DIFF_TIME(t1, t2));

	gettimeofday(&t1, NULL);
	// filering
	poly_pairs = spatial_filter(&data1->index, &data2->index);
	if(!poly_pairs)
		goto out;
	gettimeofday(&t2, NULL);
	printf("Time on filtering: %lf s\n", DIFF_TIME(t1, t2));

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
	printf("Time on refinement and spatial op: %lf s\n", DIFF_TIME(t1, t2));

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
  float *ratios = NULL;
  int count = 0;
  string tab = "\t";
  string comma = ",";
  string input_line;
  string tid ;
  string prev_tid = "";
  
  string geom_info;
  geom_meta_array[0] = new vector<string>;
  geom_meta_array[1] = new vector<string>;

  size_t pos, pos2;
  /* vector<float> rat; //collection of all ratios */
  while(cin && getline(cin, input_line) && !cin.eof()) {
    pos=input_line.find_first_of(tab,0);
    if (pos == string::npos){
      cerr << "no TAB in the input! We are toasted." << endl;
      return 1; // failure
    }

    tid= input_line.substr(0,pos); // tile id

    // finished reading in a tile data, so perform cross matching
    if (0 != tid.compare(prev_tid) && prev_tid.size()>0) 
    {
      crossmatch(ratios,count);
      geom_meta_array[0].clear();
      geom_meta_array[1].clear();
      nr_polys = 0;
      nr_vertices = 0;
    }
    // actual geometry info: did,oid,num_ver,mbb, geom
    int i = input_line[pos+1] - '1'; // array position 
    pos2=input_line.find_first_of(comma,pos+3); //oid = input_line.substr(pos+3,pos2-pos-3) 
    pos=input_line.find_first_of(comma,pos2+1); //num_ver = input_line.substr(pos2+1,pos)
    nr_vertices += atoi(input_line.substr(pos2+1,pos-pos2-1));
    nr_polys +=1;

    geom_info = input_line.substr(pos2+1); // nv,mbb,geom
    geom_meta_array[i].push_back(geom_info);
    prev_tid = tid; 
  }
  // last tile 
  crossmatch(ratios,count);

  //clear memory 
  for (int i =0; i <2; i++){
    geom_meta_array[i].clear();
    delete geom_meta_array[i];
  }

  cout.flush();
  cerr.flush();

  return 0;
}
