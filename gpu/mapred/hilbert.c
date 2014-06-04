#include <stdlib.h>
#include <math.h>
#include "hilbert.h"

static inline void mbr_center(const mbr_t *mbr, int *x, int *y){
	*x=(mbr->l+mbr->r)/2;
	*y=(mbr->b+mbr->t)/2;
}

static int compare(const void *a, const void *b){
	return (((mbr_idx_t *)a)->hbt_val - ((mbr_idx_t *)b)->hbt_val);
}

/**
 * unused
 * Interleave the bits from two input integer values
 * @param odd integer holding bit values for odd bit positions
 * @param even integer holding bit values for even bit positions
 * @return the integer that results from interleaving the input bits
 *

static int interleaveBits(int odd, int even) {
    int val = 0;
    int max = MAX(odd, even);
    int i, bitMask, a, b, n = 0;

    while (max > 0) {
      n++;
      max >>= 1;
    }

    for (i = 0; i < n; i++) {
        bitMask = 1 << i;
        a = (even & bitMask) > 0 ? (1 << (2*i)) : 0;
        b = (odd & bitMask) > 0 ? (1 << (2*i+1)) : 0;
        val += a + b;
    }

    return val;
}
 */

/* http://graphics.stanford.edu/~seander/bithacks.html#InterleaveBMN
 */
static unsigned int interleaveBits(int odd, int even) {
	static const unsigned int B[] = {0x55555555, 0x33333333, 0x0F0F0F0F, 0x00FF00FF};
	static const unsigned int S[] = {1, 2, 4, 8};

	unsigned int x=even; 	// Interleave lower 16 bits of x and y, so the bits of x
	unsigned int y=odd; 	// are in the even positions and bits from y in the odd;
	unsigned int z; 		// z gets the resulting 32-bit Morton Number.
	                		// x and y must initially be less than 65536.

	x = (x | (x << S[3])) & B[3];
	x = (x | (x << S[2])) & B[2];
	x = (x | (x << S[1])) & B[1];
	x = (x | (x << S[0])) & B[0];

	y = (y | (y << S[3])) & B[3];
	y = (y | (y << S[2])) & B[2];
	y = (y | (y << S[1])) & B[1];
	y = (y | (y << S[0])) & B[0];

	z = x | (y << 1);

	return z;
}

/**
 * Find the Hilbert value (=vertex index) for the given grid cell
 * coordinates.
 * @param x cell column (from 0)
 * @param y cell row (from 0)
 * @param r resolution of Hilbert curve (grid will have 2^r
 * rows and cols)
 * @return Hilbert value
 *
 * http://dx.doi.org/10.1002/(SICI)1097-024X(199612)26:12%3C1335::AID-SPE60%3E3.3.CO;2-1
 */
static unsigned int hbt_encode(int x, int y, int r) {

    int mask = (1 << r) - 1;
    int hodd = 0;
    int heven = x ^ y;
    int notx = ~x & mask;
    int noty = ~y & mask;
    int temp = notx ^ y;

    int k, v0 = 0, v1 = 0;
    for (k = 1; k < r; k++) {
        v1 = ((v1 & heven) | ((v0 ^ noty) & temp)) >> 1;
        v0 = ((v0 & (v1 ^ notx)) | (~v0 & (v1 ^ noty))) >> 1;
    }
    hodd = (~v0 & (v1 ^ x)) | (v0 & (v1 ^ noty));

    return interleaveBits(hodd, heven);
}

int build_spatial_index_hilbert(
	r_tree_t **index,
	mbr_t *mbrs,
	const int nr_polys)
{
	const int height=log(nr_polys)/log(MAX_NR_ENTRIES_PER_NODE)+1;
	int nr_nodes; 
	r_tree_t *rtree;
	int x, y, i, j, count;

	//indices of the first node of each level of the tree
	int level_offset[height+1];

	mbr_idx_t mbr_idx[nr_polys];

	rtree = (r_tree_t *)malloc(sizeof(r_tree_t));
	if(!rtree) {
		return -1;
	}

	// we allocate an array of r-tree nodes, which enough
	// to accommodate the mbrs to be inserted
	nr_nodes = (nr_polys / (MIN_NR_ENTRIES_PER_NODE - 1) + 2);
	rtree->nodes = (r_tree_node_t *)malloc(nr_nodes * sizeof(r_tree_node_t));
	if(!rtree->nodes) {
		free(rtree);
		return -1;
	}

	// calculate Hilbert values and store mbr-idx mapping
	for(i = 0; i < nr_polys; i++) {
		mbr_idx[i].mbr=&mbrs[i];
		mbr_idx[i].idx=i;
		mbr_center(&mbrs[i], &x, &y);
		mbr_idx[i].hbt_val=hbt_encode(x, y, HBT_RESOLUTION);
	}

	// sort leaf mbrs
	qsort(mbr_idx, nr_polys, sizeof(mbr_idx_t), compare);

	r_tree_node_t *node;
	i=0;
	count=0;
	level_offset[0]=0;
	rtree->next_free = &rtree->nodes[0];

	//fill leaf nodes
	while(i < nr_polys) {
		node=rtree->next_free;
		for(j=0; j<MAX_NR_ENTRIES_PER_NODE && i<nr_polys; j++){
			node->node_type = NODE_LEAF;
			node->entries[j].mbr = *(mbr_idx[i].mbr);
			node->entries[j].ref.idx = mbr_idx[i].idx;
			i++;
		}
		node->nr_entries=j;
		rtree->next_free++;
		count++;
	}
	level_offset[1]=count;

	//fill non-leaf nodes level by level
	int l=1;	//current level
	while(l<height){
		i=level_offset[l-1];

		while(i<level_offset[l]){
			node=rtree->next_free;
			for(j=0; j<MAX_NR_ENTRIES_PER_NODE && i<level_offset[l]; j++){
				node->node_type = NODE_NONLEAF;
				//calculate mbr
				node->entries[j].mbr = get_mbr(&rtree->nodes[i]);
				node->entries[j].ref.child = &rtree->nodes[i];
				//update parent info
				rtree->nodes[i].parent_entry = &node->entries[j];
				rtree->nodes[i].parent_node = node;
				i++;
			}
			node->nr_entries=j;
			rtree->next_free++;
			count++;
		}

		l++;
		level_offset[l]=count;
	}

	rtree->max_nr_nodes=count;
	rtree->root=&(rtree->nodes[count-1]);
	rtree->root->parent_node = NULL;
	rtree->root->parent_entry = NULL;
	*index = rtree;
	return 0;
}

void free_spatial_index_hilbert(r_tree_t *rtree)
{
	free(rtree->nodes);
	free(rtree);
}

poly_pair_array_t *spatial_filter_hilbert(
	r_tree_t *rtree1,
	r_tree_t *rtree2)
{
	return spatial_filter_r(rtree1, rtree2);
}
