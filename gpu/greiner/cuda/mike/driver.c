#include  <stdio.h> 
#include  <stdlib.h> 
#include  "spatial.cuh"

int gpuIntersect(VERTEX *poly1, int poly1Size, VERTEX *poly2, int poly2Size, VERTEX **intPoly, int *intPolySize);
bool AddVertex(vertex *&poly, int& size, vertex *vert);

bool add(int which_poly, int x, int y);
void vis(vertex* p, int size);
void test1();
void test2();
void test3();

VERTEX *s = NULL;
VERTEX *c = NULL;
VERTEX *r = NULL;
int s_size = 0;
int c_size = 0;
int r_size = 0;

int main(int argc, char **argv) 
{
    vertex *aux ; 
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
		exit(1);
	}
    else {
	printf("please provide an test arugument: [1,2,3] \n");
	exit(1);
    }

    fprintf(stderr,"s_size: %d\n",s_size);
    vis(s, s_size);

    printf("\n");
    
    fprintf(stderr,"c_size: %d\n",c_size);
    vis(c, c_size);
    
    
    gpuIntersect(s,s_size,c,c_size,&r,&r_size);
    fprintf(stderr,"\n\n");

    fprintf(stderr,"r_size: %d\n",r_size);
    vis(r,r_size);
}

void vis(vertex * p, int size ) {
    vertex * aux = p;
    while (aux->next <= size && aux->next > 0) {
	fprintf(stderr,"(%d,%d)\n",(int)aux->x,(int)aux->y);
	aux = &(p[aux->next]); 
    }
}

bool add(int which_poly, int x, int y) 
{ 
    bool res = false;
    vertex *v; 

    v = (vertex*)malloc(sizeof(vertex)); 
    v->x = x; 
    v->y = y;
    v->alpha = 0.;
    v->internal = false;
    v->linkTag = 0;

    if (which_poly == 1) 
    { 
	v->next = s_size+1 ;
	res = AddVertex(s,s_size,v);
    } 
    else if (which_poly == 2) 
    { 
	v->next = c_size+1 ;
	res = AddVertex(c,c_size,v);
    } 
    else {
	printf("%d is not a valid polygon index.\n",which_poly);
	exit(1);
    }
    free(v);

    return res;
}

void test1()
{
    if (!add(1,3,2))
	fprintf(stderr,"failed.\n");
    if (!add(1,1,2))
	fprintf(stderr,"failed.\n");
    if (!add(1,1,4))
	fprintf(stderr,"failed.\n");
    if (!add(1,3,4))
	fprintf(stderr,"failed.\n");

    if (!add(2,4,1))
	fprintf(stderr,"failed.\n");
    if (!add(2,2,1))
	fprintf(stderr,"failed.\n");
    if (!add(2,2,3))
	fprintf(stderr,"failed.\n");
    if (!add(2,4,3))
	fprintf(stderr,"failed.\n");
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

