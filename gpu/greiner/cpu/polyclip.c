#include <stdio.h> 
#include <stdlib.h> 
#include <ctype.h> 
#include <math.h> 
#include <string.h>
#include <float.h>
#include "node.h"

node *s=0, *c=0, *root=0; 
// int pS=0, pC=1; 

void view_node(node *p) 
{ 
  if(p) printf("%c%c%c (%3d,%3d)  %f    c:%10p n:%10p P:%10p\n", 
        p->intersect ? 'I' : ' ', 
        p->entry ? 'E' : ' ', 
        p->visited ? 'X' : ' ', 
        p->x, p->y, p->alpha, p, p->neighbor, p->nextPoly); 
  else  puts("NULL"); 
}

void view(node *p) 
{ 
  node *aux=p; 
  puts("");

  if(aux) do 
  { 
        view_node(aux); 
        aux=aux->next; 
  } 
  while(aux && aux != p); 
}


void deleteNode(node *p) 
{ 
  node *aux, *hold;

  if(hold=p) do 
  { 
        aux=p; 
        p=p->next; 
        free(aux); 
  } 
  while(p && p!=hold); 
}

void insert(node *ins, node *first, node *last) 
{ 
  node *aux=first; 
  while(aux != last && aux->alpha < ins->alpha) aux = aux->next; 
  ins->next = aux; 
  ins->prev = aux->prev; 
  ins->prev->next = ins; 
  ins->next->prev = ins; 
}

/* creates  a node with x y coordinates, and inserts it into the space between prev and next. 
 * prev ----> newNode ---> next 
 * */
node *create(int x, int y, node *next, node *prev, node *nextPoly, node *neighbor, int intersect, int entry, int visited, float alpha)
{ 
  node *newNode = (node*) malloc(sizeof(node)); 
  newNode->x = x; 
  newNode->y = y; 
  newNode->next = next; 
  newNode->prev = prev; 
  if(prev) newNode->prev->next = newNode; 
  if(next) newNode->next->prev = newNode; 
  newNode->nextPoly = nextPoly; 
  newNode->neighbor = neighbor; 
  newNode->intersect = intersect; 
  newNode->entry = entry; 
  newNode->visited = visited; 
  newNode->alpha = alpha; 
  return newNode; 
}

node *next_node(node *p) 
{ 
  node *aux=p; 
  while(aux && aux->intersect) aux=aux->next; 
  return aux; 
}

node *last_node(node *p) 
{ 
  node *aux=p; 
  if(aux) while(aux->next) aux=aux->next; 
  return aux; 
}

// this function finds first unprocessed and intersecting node 
node *first(node *p) 
{ 
  node *aux=p;

  if (aux) 
  do aux=aux->next; 
  while(aux!=p && (!aux->intersect || aux->intersect && aux->visited)); 
  return aux; 
}

void circle(node *p) 
{ 
  node *aux = last_node(p); 
  aux->prev->next = p; 
  p->prev = aux->prev; 
  free(aux); 
}

float dist(float x1, float y1, float x2, float y2) 
{ 
  return sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2)); 
}

/* test if the lines segments p1--p2 and q1--q2 intersects. */ 

int I(node *p1, node *p2, node *q1, node *q2, float *alpha_p, float *alpha_q, int *xint, int *yint) 
{ 
    float x, y, tp, tq, t, par ;

    par = (float)((p2->x - p1->x)*(q2->y - q1->y) - (p2->y - p1->y)*(q2->x - q1->x));
    if (! par ) return 0;                               /* parallel lines */
    // if (par < FLT_EPSILON ) return 0;                               /* parallel lines */

    tp = ((q1->x - p1->x)*(q2->y - q1->y) - (q1->y - p1->y)*(q2->x - q1->x))/par; 
    tq = ((p2->y - p1->y)*(q1->x - p1->x) - (p2->x - p1->x)*(q1->y - p1->y))/par;

    if(tp<0 || tp>1 || tq<0 || tq>1) return 0;

    x = p1->x + tp*(p2->x - p1->x); 
    y = p1->y + tp*(p2->y - p1->y);

    *alpha_p = dist(p1->x, p1->y, x, y) / dist(p1->x, p1->y, p2->x, p2->y); 
    *alpha_q = dist(q1->x, q1->y, x, y) / dist(q1->x, q1->y, q2->x, q2->y); 
    *xint = (int) x; 
    *yint = (int) y;

    return 1; 
}

/* test if the point is inside the polygon pointed by p */
int test(node *point, node *p) 
{ 
    node *aux, *left, i; 
    int type=0;

    left = create(0, point->y, 0, 0, 0, 0, 0, 0, 0, 0.); 
    for(aux=p; aux->next; aux=aux->next) 
    {
	if(I(left, point, aux, aux->next, &i.alpha, &i.alpha, &i.x, &i.y)) 
	    type++; 
    }
    return type%2; 
}

void quit() 
{ 
    deleteNode(s); 
    deleteNode(c); 
    exit(0); 
}

void clip() 
{ 
    node *auxs, *auxc, *is, *ic; 
    int xi, yi, e; 
    float alpha_s, alpha_c;

    node *crt, *newNode, *old; 
    int forward;

    if(!s || !c) return; /*if one of the polygon is empty*/

    // close the polygon by providing the first point as the last point
    auxs = last_node(s); 
    create(s->x, s->y, 0, auxs, 0, 0, 0, 0, 0, 0.); 
    auxc = last_node(c); 
    create(c->x, c->y, 0, auxc, 0, 0, 0, 0, 0, 0.);

    // phase 1
    for(auxs = s; auxs->next; auxs = auxs->next) 
    {
	if(!auxs->intersect) 
	{
	    for(auxc = c; auxc->next; auxc = auxc->next) 
	    {
		if(!auxc->intersect) 
		{
		    if(I(auxs, next_node(auxs->next), auxc, next_node(auxc->next), &alpha_s, &alpha_c, &xi, &yi)) 
		    { 
			is = create(xi, yi, 0, 0, 0, 0, 1, 0, 0, alpha_s); 
			ic = create(xi, yi, 0, 0, 0, 0, 1, 0, 0, alpha_c); 
			is->neighbor = ic; 
			ic->neighbor = is; 
			insert(is, auxs, next_node(auxs->next)); 
			insert(ic, auxc, next_node(auxc->next)); 
		    }
		}
	    }
	}
    }

    // phase 2
    /*determine the exit point of the polygon using odd-even rule*/
    e = test(s, c); 
    e = 1-e; 

    for(auxs = s; auxs->next; auxs = auxs->next)
    {
	if(auxs->intersect) 
	{ 
	    auxs->entry = e; 
	    e = 1-e; 
	}
    }

    e=test(c, s); 
    e = 1-e;

    for(auxc = c; auxc->next; auxc = auxc->next) 
    {
	if(auxc->intersect) 
	{ 
	    auxc->entry = e; 
	    e = 1-e; 
	}
    }
    /* delete last node and make the polygon list circular */
    circle(s); 
    circle(c);

    view(s); 
    view(c);
    // phase 3 
    while ((crt = first(s)) != s) 
    { 
	old = 0; 
	for(; !crt->visited; crt = crt->neighbor) 
	    for(forward = crt->entry ;; ) 
	    { 
		printf("(%d,%d)\t",crt->x,crt->y);
		newNode = create(crt->x, crt->y, old, 0, 0, 0, 0, 0, 0, 0.); 
		old = newNode; 
		crt->visited = 1; 
		crt = forward ? crt->next : crt->prev; 
		if(crt->intersect) 
		{ 
		    crt->visited = 1; 
		    break; 
		} 
	    }

	old->nextPoly = root; 
	root = old; 
	printf("\n");
    }

    view(s); 
    view(c);

}

void add(int which_poly, int x, int y) 
{ 
    node *newNode; 

    newNode = (node*)malloc(sizeof(node)); 
    newNode->x = x; 
    newNode->y = y; 
    newNode->prev = 0;        /* not need to initialize with 0 after malloc ... */ 
    newNode->nextPoly = 0; 
    newNode->neighbor = 0; 
    newNode->intersect = 0; 
    newNode->entry = 0; 
    newNode->visited = 0; 
    newNode->alpha = 0.; 
    if (which_poly == 1) 
    { 
	newNode->next = s; 
	if (s) s->prev = newNode; 
	s = newNode; 
    } 
    else if (which_poly == 2) 
    { 
	newNode->next = c; 
	if (c) c->prev = newNode; 
	c = newNode; 
    } 
    else {
	deleteNode(newNode);
	printf("%d is not a valid polygon index.\n",which_poly);
	quit();
    }
}

void result(){
    node *aux=root; 
    while (root){
	while (aux){
	    printf("(%d,%d)\t",aux->x,aux->y);
	    aux = aux->next;
	}
	printf("\n");
	root = root->nextPoly; 
    }
}

void test1();
void test2();
void test3();

int main(int argc, char **argv) 
{
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
	    case '4':
		break;
	    default:
		;
	}
    clip();
    //result();
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

