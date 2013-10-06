#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <sys/time.h>
#include "gpu/gpu_spatial.h"
#include "tbb/parallel_for.h"
#include "tbb/blocked_range.h"
#include "buffers.h"
#include "constants.h"

using namespace tbb;

// input buffer
file_names_buffer buf_file_names;
// output buffer
extern dequeue<poly_arrays_buffer_item> buf_poly_arrays;
extern dequeue<poly_pairs_buffer_item> buf_poly_pairs;

// how to parse task
const int nr_lines_per_task_min = 100;
const int nr_parser_tasks_max = nr_procs;

// the temp poly buffer for each parser task
poly_array_t sub_polys[nr_parser_tasks_max];


int poly_array_malloc_size(const int nr_polys, const int nr_vertices)
{
	// to optimize memory layout on cpu/gpu, we allocate a large continuous space
	// that accommodates mbrs, offsets, x and y arrays together; in this manner,
	// only one memory movement is needed to transfer all these data from cpu
	// to gpu
	int size_mbrs = nr_polys * sizeof(mbr_t);
	int size_offsets = (nr_polys + 1) * sizeof(int);
	int size_x = nr_vertices * sizeof(int);
	int size_y = nr_vertices * sizeof(int);

	return size_mbrs + size_offsets + size_x + size_y;
}

int alloc_poly_array(poly_array_t *polys, const int nr_polys, const int nr_vertices)
{
	int size_mbrs = nr_polys * sizeof(mbr_t);
	int size_offsets = (nr_polys + 1) * sizeof(int);
	int size_x = nr_vertices * sizeof(int);
	int size_y = nr_vertices * sizeof(int);

	polys->mbrs = (mbr_t *)malloc(size_mbrs + size_offsets + size_x + size_y);
	if(!polys->mbrs) {
		fprintf(stderr, "failed to allocate memory for poly array\n");
		exit(1);
	}

	polys->nr_polys = nr_polys;
	polys->nr_vertices = nr_vertices;
	polys->offsets = (int *)((char *)(polys->mbrs) + size_mbrs);
	polys->x = (int *)((char *)(polys->offsets) + size_offsets);
	polys->y = (int *)((char *)(polys->x) + size_x);

	return 0;
}

static int adjust_poly_array(poly_array_t *polys, const int nr_polys, const int nr_vertices)
{
	int size_mbrs = nr_polys * sizeof(mbr_t);
	int size_offsets = (nr_polys + 1) * sizeof(int);
	int size_x = nr_vertices * sizeof(int);

	polys->nr_polys = nr_polys;
	polys->nr_vertices = nr_vertices;
	polys->offsets = (int *)((char *)(polys->mbrs) + size_mbrs);
	polys->x = (int *)((char *)(polys->offsets) + size_offsets);
	polys->y = (int *)((char *)(polys->x) + size_x);
}

int realloc_poly_array(poly_array_t *polys, const int nr_polys, const int nr_vertices)
{
	int size_mbrs = nr_polys * sizeof(mbr_t);
	int size_offsets = (nr_polys + 1) * sizeof(int);
	int size_x = nr_vertices * sizeof(int);
	int size_y = nr_vertices * sizeof(int);

	FREE(polys->mbrs);
	polys->mbrs = (mbr_t *)malloc(size_mbrs + size_offsets + size_x + size_y);
	if(!polys->mbrs) {
		fprintf(stderr, "failed to re-allocate memory for poly array\n");
		return -1;
	}

	polys->nr_polys = nr_polys;
	polys->nr_vertices = nr_vertices;
	polys->offsets = (int *)((char *)(polys->mbrs) + size_mbrs);
	polys->x = (int *)((char *)(polys->offsets) + size_offsets);
	polys->y = (int *)((char *)(polys->x) + size_x);

	return 0;
}


class PolyParser
{
	static const int parse_buf_size = 8192;
	FILE *file;
	size_t nr_lines_per_task;

	void parselines(poly_array_t *polys) const {
		char *p, *q;
		int offset = 0;
		int ipoly = 0;
		int nr_lines_parsed = 0;
		char *parse_buf = (char *)malloc(parse_buf_size);

		if(!parse_buf) {
			fprintf(stderr, "failed to allcoate parse buffer\n");
			exit(1);
		}

		// read and parse each text line
		while((nr_lines_parsed++) < nr_lines_per_task && fgets(parse_buf, parse_buf_size, file)) {
			if(parse_buf[0] == '\n' || parse_buf[0] == '\0')
				continue;
			polys->offsets[ipoly] = offset;

			// omit prefix chars until the first ','
			p = parse_buf;
			while(*p != ',') p++;
			p++;

			// read in mbr data
			sscanf(p, " %d %d %d %d", &polys->mbrs[ipoly].l, &polys->mbrs[ipoly].r,
				&polys->mbrs[ipoly].b, &polys->mbrs[ipoly].t);

			// omit mbr text
			while(*p != ',') p++;
			p++;

			// parse vertex data
			do {
				q = p;
				while(*q != ',') q++;
				sscanf(p, " %d %d", &polys->x[offset], &polys->y[offset]);
				p = q + 1;
				offset++;
			} while(*p != '\n' && *p != '\0');

			// move on to the next poly
			ipoly++;
		}

		polys->offsets[ipoly] = offset;
		polys->nr_polys=ipoly;
		polys->nr_vertices=offset;
		free(parse_buf);
	}

public:
	void operator()(const blocked_range<int>& r) const
	{
		for(int i = r.begin(); i != r.end(); i++) {
			parselines(&sub_polys[i]);
		}
	}

	PolyParser(FILE *fp, size_t nr_lines_per) :
		file(fp), nr_lines_per_task(nr_lines_per) {}
};

poly_array_t *load_and_parse_polys(const char *file_name)
{
	static int poly_array_malloc_size_max = 0;
	static int last_nr_polys = 0, last_nr_vertices = 0;
	poly_array_t *polys = NULL;
	FILE *file = NULL;
	int nr_tasks, nr_lines_per_task;
	int new_size;
	int nr_polys, nr_vertices;
	char readbuf[64];
	int n_poly, n_vt;
	char *mbrs_head, *x_head, *y_head;
	int sum_vt=0, offsets_index=0;

	file = fopen(file_name, "r");
	if(!file) {
		fprintf(stderr, "failed to open file: %s\n", file_name);
		goto failure;
	}

	// read the first line: number of polygons and total number of vertices
	if(!fgets(readbuf, 63, file)) {
		goto failure;
	}
	sscanf(readbuf, "%d, %d\n", &nr_polys, &nr_vertices);

	// how many tasks to spawn and the max number of lines parsed by each task
	nr_tasks = nr_polys / nr_lines_per_task_min;
	nr_tasks = (nr_polys % nr_lines_per_task_min) ? (nr_tasks + 1) : nr_tasks;
	if(nr_tasks > nr_parser_tasks_max)
		nr_tasks = nr_parser_tasks_max;
	nr_lines_per_task = nr_polys / nr_tasks;
	nr_lines_per_task = (nr_polys % nr_tasks) ? (nr_lines_per_task + 1) : nr_lines_per_task;

	// allocate/adjust memory for intermediate poly array buffers
	new_size = poly_array_malloc_size(nr_polys, nr_vertices);
	if(new_size > poly_array_malloc_size_max) {
		poly_array_malloc_size_max = new_size;
		last_nr_polys = nr_polys;
		last_nr_vertices = nr_vertices;
		for(int i = 0; i < nr_parser_tasks_max; i++) {
			realloc_poly_array(&sub_polys[i], nr_polys, nr_vertices);
		}
	}
	else if(nr_polys > last_nr_polys || nr_vertices > last_nr_vertices) {
		last_nr_polys = nr_polys;
		last_nr_vertices = nr_vertices;
		for(int i = 0; i < nr_parser_tasks_max; i++) {
			adjust_poly_array(&sub_polys[i], nr_polys, nr_vertices);
		}
	}

	// begin parallel parsing
	parallel_for(blocked_range<int>(0, nr_tasks), PolyParser(file, nr_lines_per_task));

	// allocate the final poly array and merge intermediate polys together
	polys = (poly_array_t *)malloc(sizeof(poly_array_t));
	if(!polys) {
		fprintf(stderr, "new poly_array_t failed\n");
		goto failure;
	}
	alloc_poly_array(polys, nr_polys, nr_vertices);

	mbrs_head=(char *)(polys->mbrs);
	x_head=(char *)(polys->x);
	y_head=(char *)(polys->y);

	for(int i = 0; i < nr_tasks; i++){
		n_poly = sub_polys[i].nr_polys;
		n_vt = sub_polys[i].nr_vertices;

		memcpy(mbrs_head, sub_polys[i].mbrs, sizeof(mbr_t)*n_poly);
		mbrs_head += sizeof(mbr_t)*n_poly;

		memcpy(x_head, sub_polys[i].x, sizeof(int)*n_vt);
		x_head += sizeof(int)*n_vt;

		memcpy(y_head, sub_polys[i].y, sizeof(int)*n_vt);
		y_head += sizeof(int)*n_vt;

		//recalculate offsets
		for(int j=0; j<n_poly; j++){
			polys->offsets[offsets_index] = sub_polys[i].offsets[j] + sum_vt;
			offsets_index++;
		}
		sum_vt += n_vt;
	}

	// the last offset indexes beyond the end of x,y arrays
	polys->offsets[offsets_index] = sum_vt;

	goto out;

failure:
	if(polys) {
		free_poly_array(polys);
		polys = NULL;
	}

out:
	if(file)
		fclose(file);
	return polys;
}

void *migrator_parser(void *param);

void *thread_parser(void *param)
{
	file_names_buffer_item file_names;
	poly_arrays_buffer_item poly_arrays;
	poly_array_t *polys_1, *polys_2;
	pthread_t tid;
	struct timeval t1, t2;
	double tot_time = 0.0;
	long n = 0;
	int state;

	// spawn the migrator thread
	if(pthread_create(&tid, NULL, migrator_parser, param)) {
		fprintf(stderr, "failed to create migration thread for the parser\n");
		exit(1);
	}

	// initialize intermediate parsing buffers
	for(int i = 0; i < nr_parser_tasks_max; i++) {
		sub_polys[i].mbrs = NULL;
	}

	while(true) {
		state = buf_file_names.pull_task(&file_names);

		if(state == 0) {
			gettimeofday(&t1, NULL);

			// load and parse file1
			polys_1 = load_and_parse_polys(file_names.file_name_1);
			if(!polys_1) {
				fprintf(stderr, "failed to parse file 1\n");
				continue;
			}

			// load and parse file2
			polys_2 = load_and_parse_polys(file_names.file_name_2);
			if(!polys_2) {
				fprintf(stderr, "failed to parse file 2\n");
				free_poly_array(polys_1);
				continue;
			}

			gettimeofday(&t2, NULL);
			tot_time += DIFF_TIME(t1, t2);
			n++;

			// push to poly_arrays buffer
			poly_arrays.polys_1 = polys_1;
			poly_arrays.polys_2 = polys_2;
			buf_poly_arrays.push_task(&poly_arrays);
		}
		else {
			buf_file_names.signal_exit();	// make sure diverter exits
			pthread_join(tid, NULL);
			buf_poly_arrays.signal_exit();
			break;
		}
	}

	printf("[parser] %d files parsed, average time per parsing: %lf s\n", n, n > 0 ? tot_time / n : 0.0);

	// delete intermediate parsing buffers
	for(int i = 0; i < nr_parser_tasks_max; i++) {
		fini_poly_array(&sub_polys[i]);
	}

	return NULL;
}

void *migrator_parser(void *param)
{
	int nr_gpus = *(int *)param;
	int dno = 0;
	file_names_buffer_item file_names;
	poly_arrays_buffer_item poly_arrays;
	poly_array_t *polys_1, *polys_2;
	struct timeval t1, t2;
	double tot_time = 0.0;
	long n = 0;
	int state;

//return NULL;

	while(true) {
		buf_file_names.lock();

repeat:
		if(buf_file_names.exiting) {
			buf_file_names.unlock();
			break;
		}

		if(buf_poly_pairs.is_empty()) {
			state = buf_file_names.pull_task_nolock(&file_names);
			buf_file_names.unlock();

			if(state == 0) {
//				printf("diverting workload from CPU\n");
				gettimeofday(&t1, NULL);

				polys_1 = gpu_parse(dno, file_names.file_name_1);
				if(!polys_1) {
					fprintf(stderr, "failed to parse file 1 on GPUs\n");
					continue;
				}

				dno = (dno + 1) % nr_gpus;

				// load and parse file2
				polys_2 = gpu_parse(dno, file_names.file_name_2);
				if(!polys_2) {
					fprintf(stderr, "failed to parse file 2 on GPUs\n");
					free_poly_array(polys_1);
					continue;
				}

				dno = (dno + 1) % nr_gpus;

				gettimeofday(&t2, NULL);
				tot_time += DIFF_TIME(t1, t2);
				n++;

				// push to poly_arrays buffer
				poly_arrays.polys_1 = polys_1;
				poly_arrays.polys_2 = polys_2;
				buf_poly_arrays.push_task(&poly_arrays);
			}
			else {
				break;
			}
		}
		else {
			buf_file_names.wait_for_congestion();
			goto repeat;
		}
	}

	printf("[parse-migrator] %d tasks processed, average time per migration: %lf s\n", n, n > 0 ? tot_time / n : 0.0);
	return NULL;
}
