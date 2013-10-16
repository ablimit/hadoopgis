#include  <stdio.h> 
#include  <stdlib.h> 
#include  "spatial.cuh"

int gpuIntersect(VERTEX *poly1, int poly1Size, VERTEX *poly2, int poly2Size, VERTEX **intPoly, int *intPolySize);
bool AddVertex(vertex *&poly, int& size, vertex *vert);

void add(int which_poly, int x, int y);
void test1();
void test2();
void test3();

VERTEX *s = NULL;
VERTEX *c = NULL;
VERTEX *r = NULL;
int s_size =-1;
int c_size =-1;
int r_size = 0;

int main(int argc, char **argv) 
{
    vertex aux ; 
    if (argc>1 )
	switch (argv[1][0]){
	    case '1':
		test1();
		break;
	    case '2':
		test2();
		break;
	    case '3':
		test3();
		break;
	    default:
		;
	}
    printf("start..\n");
    gpuIntersect(s,s_size,c,c_size,&r,&r_size);
    printf("end..\n");
    aux = *r;
    for (int i=0; i< r_size; i++ ){
	    printf("(%f,%f)\t",aux.x,aux.y);
	    aux = r[aux.next];
    }
}

void add(int which_poly, int x, int y) 
{ 
    vertex *v; 

    v = (vertex*)malloc(sizeof(vertex)); 
    v->x = x; 
    v->y = y;
    v->alpha = 0.;
    v->next =0;
    v->internal =false;
    v->linkTag=0;

    if (which_poly == 1) 
    { 
	AddVertex(s,s_size,v);
	s_size++;
    } 
    else if (which_poly == 2) 
    { 
	AddVertex(c,c_size,v);
	c_size++; 
    } 
    else {
	printf("%d is not a valid polygon index.\n",which_poly);
	exit(1);
    }
    free(v);
}

void test1()
{
    add(1,3,2);
    add(1,1,2);
    add(1,1,4);
    add(1,3,4);

    add(2,4,1);
    add(2,2,1);
    add(2,2,3);
    add(2,4,3);
}

void test2()
{
    add(2,7,3);
    add(2,1,3);
    add(2,1,6);
    add(2,7,6);

    add(1,6,1);
    add(1,2,1);
    add(1,2,4);
    add(1,4,2);
    add(1,6,4);
}

void test3()
{
    add(2,4,4);
    add(2,1,4);
    add(2,1,1);
    add(2,4,1);

    add(1,3,3);
    add(1,2,3);
    add(1,2,2);
    add(1,3,2);
}

