#ifndef RTREE_H
#define RTREE_H

#include "spatial.h"

#define MAX_NR_ENTRIES_PER_NODE		60		// M
#define MIN_NR_ENTRIES_PER_NODE		30		// m

#define ABS(x)	((x) < 0 ? -(x) : (x))

// r-tree node type
typedef enum
{
	NODE_LEAF,
	NODE_NONLEAF
} r_tree_node_type_t;

struct r_tree_node_struct;
// index entry
typedef struct index_entry_struct
{
	mbr_t mbr;
	union {
		struct r_tree_node_struct *child;	// child pointer if nonleef node
		int idx;	// index into polygon array if leef node
	} ref;
} index_entry_t;

// polygon pairs block: temporary storage when constructing
// polygon pairs array
#define POLY_PAIR_BLOCK_SIZE	256
typedef struct poly_pair_block_struct
{
	int nr_poly_pairs;
	mbr_t mbrs[POLY_PAIR_BLOCK_SIZE];	// mbrs of each polygon pair
	int idx1[POLY_PAIR_BLOCK_SIZE];		// index to the first of each polygon pair
	int idx2[POLY_PAIR_BLOCK_SIZE];		// index to the second of each polygon pair
	struct poly_pair_block_struct *next_block;
} poly_pair_block_t;

// r-tree node
typedef struct r_tree_node_struct
{
	r_tree_node_type_t node_type;
	int nr_entries;
	struct r_tree_node_struct *parent_node;
	index_entry_t *parent_entry;
	index_entry_t entries[MAX_NR_ENTRIES_PER_NODE];
} r_tree_node_t;

// r-tree
typedef struct r_tree_struct
{
	r_tree_node_t *root;		// the root
	r_tree_node_t *next_free;	// next free node
	int max_nr_nodes;			// maximum nr of nodes in the nodes array; unused
	r_tree_node_t *nodes;		// the nodes array
} r_tree_t;


int build_spatial_index_r(
	r_tree_t **index,
	mbr_t *mbrs,
	int nr_polys);
void free_spatial_index_r(r_tree_t *rtree);
poly_pair_array_t *spatial_filter_r(
	r_tree_t *rtree1,
	r_tree_t *rtree2);

/*******************************************************************************
 * R-Tree routines
 */
// local procedure declarations
void pick_seeds(index_entry_t **entries, int *left, int *right);
int pick_next(
    index_entry_t **entries,
    int begin, int end,
    mbr_t *mbr_left, mbr_t *mbr_right);
void adjust_tree_r(
    r_tree_t *rtree,
    r_tree_node_t *node,
    mbr_t *mbr_inc,
    r_tree_node_t *new_node);
r_tree_node_t *split_node_r(
    r_tree_t *rtree,
    r_tree_node_t *node,
    mbr_t *mbr,
    void *inserted);
r_tree_node_t *choose_leaf_r(r_tree_t *rtree, mbr_t *mbr);
inline void set_leaf_entry(index_entry_t *entry, mbr_t *mbr, int idx);
inline void set_nonleaf_entry(
    index_entry_t *entry,
    mbr_t *mbr,
    r_tree_node_t *child);
void insert_r(r_tree_t *rtree, mbr_t *mbr, int idx);
int spatial_filter_r_1(
    r_tree_node_t *node1,
    r_tree_node_t *node2,
    poly_pair_block_t **first_block,
    int *nr_poly_pairs);
int spatial_filter_r_2(
    r_tree_node_t *node1,
    index_entry_t *leaf_entry,
    poly_pair_block_t **first_block,
    int *nr_poly_pairs);
int spatial_filter_r_3(
    index_entry_t *leaf_entry,
    r_tree_node_t *node2,
    poly_pair_block_t **first_block,
    int *nr_poly_pairs);
inline int filter_callback_r(
    mbr_t *mbr1, int idx1,
    mbr_t *mbr2, int idx2,
    poly_pair_block_t **first_block,
    int *nr_poly_pairs);
mbr_t get_mbr(r_tree_node_t *node);

#endif
