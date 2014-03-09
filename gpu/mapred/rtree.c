#include <stdlib.h>
#include <string.h>
#include "rtree.h"

// TODO: Optimizations
// 1. the layout of tree nodes in the nodes array
// 2. 

/*******************************************************************************
 * R-Tree routines
 */
// local procedure declarations
static void pick_seeds(const index_entry_t **entries, int *left, int *right);
static int pick_next(
	const index_entry_t **entries,
	int begin, int end,
	const mbr_t *mbr_left, const mbr_t *mbr_right);
static void adjust_tree_r(
	r_tree_t *rtree,
	r_tree_node_t *node,
	mbr_t *mbr_inc,
	r_tree_node_t *new_node);
static r_tree_node_t *split_node_r(
	r_tree_t *rtree,
	r_tree_node_t *node,
	mbr_t *mbr,
	void *inserted);
static r_tree_node_t *choose_leaf_r(r_tree_t *rtree, const mbr_t *mbr);
static inline void set_leaf_entry(index_entry_t *entry, const mbr_t *mbr, const int idx);
static inline void set_nonleaf_entry(
	index_entry_t *entry,
	const mbr_t *mbr,
	const r_tree_node_t *child);
static void insert_r(r_tree_t *rtree, const mbr_t *mbr, int idx);
static int spatial_filter_r_1(
	const r_tree_node_t *node1,
	const r_tree_node_t *node2,
	poly_pair_block_t **first_block,
	int *nr_poly_pairs);
static int spatial_filter_r_2(
	const r_tree_node_t *node1,
	const index_entry_t *leaf_entry,
	poly_pair_block_t **first_block,
	int *nr_poly_pairs);
static int spatial_filter_r_3(
	const index_entry_t *leaf_entry,
	const r_tree_node_t *node2,
	poly_pair_block_t **first_block,
	int *nr_poly_pairs);
static inline int filter_callback_r(
	const mbr_t *mbr1, const int idx1,
	const mbr_t *mbr2, const int idx2,
	poly_pair_block_t **first_block,
	int *nr_poly_pairs);
mbr_t get_mbr(const r_tree_node_t *node);


// get the mbr covering all entries in a tree node
mbr_t get_mbr(const r_tree_node_t *node)
{
	mbr_t mbr = node->entries[0].mbr;
	int i;

	for(i = 1; i < node->nr_entries; i++) {
		mbr_update(&mbr, &node->entries[i].mbr);
	}

	return mbr;
}

// This function is unused
// NOTE: what if chosen left and right are the same?
static void pick_seeds_linear(index_entry_t **entries, int *left, int *right)
{
	int hls_x, lhs_x, hls_y, lhs_y;
	int i_hls_x, i_lhs_x, i_hls_y, i_lhs_y;
	int min_x, max_x, min_y, max_y;
	int i;

	// get the lowest high side and highest low side of each dimension
	hls_x = hls_y = A_VERY_SMALL_NUM;
	lhs_x = lhs_y = A_VERY_LARGE_NUM;
	min_x = min_y = A_VERY_LARGE_NUM;
	max_x = max_y = A_VERY_SMALL_NUM;
	for(i = 0; i <= MAX_NR_ENTRIES_PER_NODE; i++) {
		if(entries[i]->mbr.l > hls_x) {
			hls_x = entries[i]->mbr.l;
			i_hls_x = i;
		}
		if(entries[i]->mbr.r < lhs_x) {
			lhs_x = entries[i]->mbr.r;
			i_lhs_x = i;
		}
		if(entries[i]->mbr.b > hls_y) {
			hls_y = entries[i]->mbr.b;
			i_hls_y = i;
		}
		if(entries[i]->mbr.t < lhs_y) {
			lhs_y = entries[i]->mbr.t;
			i_lhs_y = i;
		}
		if(entries[i]->mbr.l < min_x) {
			min_x = entries[i]->mbr.l;
		}
		if(entries[i]->mbr.r > max_x) {
			max_x = entries[i]->mbr.r;
		}
		if(entries[i]->mbr.b < min_y) {
			min_y = entries[i]->mbr.b;
		}
		if(entries[i]->mbr.t > max_y) {
			max_y = entries[i]->mbr.t;
		}
	}

	// normalize and decide which pair to choose as the seeds
	if((ABS(hls_x - lhs_x) / (double)(max_x - min_x)) >
		(ABS(hls_y - lhs_y) / (double)(max_y - min_y))) {
		*left = i_hls_x;
		*right = i_lhs_x;
	}
	else {
		*left = i_hls_y;
		*right = i_lhs_y;
	}
}

static void pick_seeds(const index_entry_t **entries, int *left, int *right)
{
	int ei, ej, d, dij;
	int ei_area, ej_area;
	int i, j;

	d = A_VERY_SMALL_NUM;

	for(i = 0; i < MAX_NR_ENTRIES_PER_NODE; i++) {
		ei_area = mbr_area(&entries[i]->mbr);

		for(j = i + 1; j < MAX_NR_ENTRIES_PER_NODE; j++) {
			ej_area = mbr_area(&entries[j]->mbr);

			dij = mbr_area_2(&entries[i]->mbr, &entries[j]->mbr)
				- ei_area - ej_area;
			if(dij > d) {
				d = dij;
				ei = i;
				ej = j;
			}
		}
	}

	*left = ei;
	*right = ej;
}

static int pick_next(
	const index_entry_t **entries,
	int begin, int end,
	const mbr_t *mbr_left, const mbr_t *mbr_right)
{
	int i, d, d_max, i_d_max;

	d_max = A_VERY_SMALL_NUM;

	for(i = begin; i < end; i++) {
		d = mbr_area_inc(mbr_left, &entries[i]->mbr) -
			mbr_area_inc(mbr_right, &entries[i]->mbr);
		d = ABS(d);
		if(d > d_max) {
			d_max = d;
			i_d_max = i;
		}
	}

	return i_d_max;
}

static void adjust_tree_r(
	r_tree_t *rtree,			// the rtree
	r_tree_node_t *node,		// the node where update happened
	mbr_t *mbr_inc,				// the mbr of the inserted if new_node is null
	r_tree_node_t *new_node)	// the new_node splitted from node if not null
{
	// if no splitting happened, simply update all ancesters' mbrs
	if(!new_node) {
		while(node->parent_entry) {
			if(mbr_update(&node->parent_entry->mbr, mbr_inc)) {
				mbr_inc = &node->parent_entry->mbr;
				node = node->parent_node;
			}
			else {
				break;
			}
		}
		return;
	}

	// if splitting happened, first update node's ancesters' mbrs
	r_tree_node_t *child_node = node;
	mbr_t mbr_new;
	while(child_node->parent_entry) {
		mbr_new = get_mbr(child_node);
		if(mbr_diff(&child_node->parent_entry->mbr, &mbr_new)) {
			child_node->parent_entry->mbr = mbr_new;
			child_node = child_node->parent_node;
		}
		else {
			break;
		}
	}

	// then propagate splitting upward
	mbr_new = get_mbr(new_node);
	if(node->parent_node) {
		node = node->parent_node;

		// if the parent has enough free space, insert the new node right there
		if(node->nr_entries < MAX_NR_ENTRIES_PER_NODE) {
			new_node->parent_node = node;
			new_node->parent_entry = &node->entries[(node->nr_entries)++];
			set_nonleaf_entry(new_node->parent_entry, &mbr_new, new_node);
			new_node = NULL;
		}

		// otherwise, split the parent node
		else {
			new_node = split_node_r(rtree, node, &mbr_new, (void *)new_node);
		}

		// recursively adjust the tree
		adjust_tree_r(rtree, node, &mbr_new, new_node);
	}
	else {
		// the root node has been splitted; a new root needs to be created
		r_tree_node_t *new_root = (rtree->next_free)++;
		new_root->node_type = NODE_NONLEAF;
		new_root->nr_entries = 2;
		new_root->parent_node = NULL;
		new_root->parent_entry = NULL;
		rtree->root = new_root;

		// insert the first split
		new_root->entries[0].mbr = get_mbr(node);
		new_root->entries[0].ref.child = node;
		node->parent_node = new_root;
		node->parent_entry = &new_root->entries[0];

		// insert the second split
		new_root->entries[1].mbr = mbr_new;
		new_root->entries[1].ref.child = new_node;
		new_node->parent_node = new_root;
		new_node->parent_entry = &new_root->entries[1];
	}
}

static r_tree_node_t *split_node_r(
	r_tree_t *rtree,		// the rtree
	r_tree_node_t *node,	// the node being splitted
	mbr_t *mbr,				// the mbr of being inserted
	void *inserted)			// the being inserted
{
	index_entry_t **entries, *entry_temp;
	index_entry_t *right_insert, *right_insert_end;
	index_entry_t *ent_to_right, *ent_to_left;
	index_entry_t *left_margin_beg, *left_margin_end;
	mbr_t mbr_left, mbr_right;
	int i, i_left, i_right;

	r_tree_node_t *new_node = (rtree->next_free)++;

	// we use the last entry of the new node to temporarily store the
	// new entry being inserted
	new_node->entries[MAX_NR_ENTRIES_PER_NODE - 1].mbr = *mbr;

	// if a leaf node is being splitted
	if(node->node_type == NODE_LEAF) {
		new_node->node_type = NODE_LEAF;
		new_node->entries[MAX_NR_ENTRIES_PER_NODE - 1].ref.idx =
			*(int *)inserted;
	}

	// if a nonleaf node is being splitted
	else {
		new_node->node_type = NODE_NONLEAF;
		new_node->entries[MAX_NR_ENTRIES_PER_NODE - 1].ref.child =
			(r_tree_node_t *)inserted;
	}

	// create a temporary array of entry indexes, which
	// will be partitioned into two subarrays and used
	// to split all entries between node and new_node
	entries = malloc((MAX_NR_ENTRIES_PER_NODE + 1) * sizeof(index_entry_t *));
	if(!entries) {
		free(new_node);
		return NULL;
	}

	for(i = 0; i < MAX_NR_ENTRIES_PER_NODE; i++) {
		entries[i] = &node->entries[i];
	}
	entries[MAX_NR_ENTRIES_PER_NODE] =
		&new_node->entries[MAX_NR_ENTRIES_PER_NODE - 1];

	// pick seeds and make initial partitions
	// [0, i_left] is the left partition; [i_right, MAX_NR_ENTRIES_PER_NODE] is
	// the second partition
	pick_seeds(entries, &i_left, &i_right);

	if(i_left > 0) {
		entry_temp = entries[i_left];
		entries[i_left] = entries[0];
		entries[0] = entry_temp;
		i_left = 0;
	}
	mbr_left = entries[0]->mbr;

	if(i_right < MAX_NR_ENTRIES_PER_NODE) {
		entry_temp = entries[i_right];
		entries[i_right] = entries[MAX_NR_ENTRIES_PER_NODE];
		entries[MAX_NR_ENTRIES_PER_NODE] = entry_temp;
		i_right = MAX_NR_ENTRIES_PER_NODE;
	}
	mbr_right = entries[MAX_NR_ENTRIES_PER_NODE]->mbr;

	// assign the rest entry indexes one by one, until all indexes have been
	// distributed or one of the partitions has
	// (MAX_NR_ENTRIES_PER_NODE - MIN_NR_ENTRIES_PER_NODE + 1) indexes.
	while(i_left < i_right - 1) {
		// if the left partition has
		// (MAX_NR_ENTRIES_PER_NODE - MIN_NR_ENTRIES_PER_NODE + 1) indexes,
		// add all the rest nodes to the right partition and finish.
		if(i_left == MAX_NR_ENTRIES_PER_NODE - MIN_NR_ENTRIES_PER_NODE) {
			i_right = i_left + 1;
		}

		// if the right partition has
		// (MAX_NR_ENTRIES_PER_NODE - MIN_NR_ENTRIES_PER_NODE + 1) indexes,
		// add all the rest nodes to the left partition and finish.
		else if(i_right == MIN_NR_ENTRIES_PER_NODE) {
			i_left = i_right - 1;
		}

		// otherwise, pick the next entry index and assign it to the partition
		// whose mbr will be enlarged the least
		else {
			i = pick_next(entries, i_left + 1, i_right, &mbr_left, &mbr_right);

			if(mbr_area_inc(&mbr_left, &entries[i]->mbr) <
				mbr_area_inc(&mbr_right, &entries[i]->mbr)) {
				if(i > ++i_left) {
					entry_temp = entries[i_left];
					entries[i_left] = entries[i];
					entries[i] = entry_temp;
				}
				mbr_update(&mbr_left, &entries[i_left]->mbr);
			}
			else {
				if(i < --i_right) {
					entry_temp = entries[i_right];
					entries[i_right] = entries[i];
					entries[i] = entry_temp;
				}
				mbr_update(&mbr_right, &entries[i_right]->mbr);
			}
		}
	}

	// we have partitioned the MAX_NR_ENTRIES_PER_NODE+1 entry indexes into two
	// sets; now use this partition to split entries to node and new_node.
	// all entries indexed by [0, i_left] will be assigned to node,
	// and all entries indexed by [i_right, MAX_NR_ENTRIES_PER_NODE] will be
	// assigned to new_node.
	right_insert = &new_node->entries[0];
	right_insert_end = right_insert + MAX_NR_ENTRIES_PER_NODE - i_right + 1;
	left_margin_beg = &node->entries[0] - 1;
	left_margin_end = &node->entries[i_left + 1];

	node->nr_entries = i_left + 1;
	new_node->nr_entries = MAX_NR_ENTRIES_PER_NODE - i_right + 1;

	if(node->node_type == NODE_LEAF) {
		while(right_insert < right_insert_end) {
			ent_to_right = entries[i_right++];
			*(right_insert++) = *ent_to_right;
			if(ent_to_right > left_margin_beg && ent_to_right < left_margin_end) {
				do {
					ent_to_left = entries[i_left--];
				} while(ent_to_left > left_margin_beg && ent_to_left < left_margin_end);
				*ent_to_right = *ent_to_left;
			}
		}
	}
	else {
		while(right_insert < right_insert_end) {
			ent_to_right = entries[i_right++];
			ent_to_right->ref.child->parent_node = new_node;
			ent_to_right->ref.child->parent_entry = right_insert;
			*(right_insert++) = *ent_to_right;
			if(ent_to_right > left_margin_beg && ent_to_right < left_margin_end) {
				do {
					ent_to_left = entries[i_left--];
				} while(ent_to_left > left_margin_beg && ent_to_left < left_margin_end);
				ent_to_left->ref.child->parent_node = node;
				ent_to_left->ref.child->parent_entry = ent_to_right;
				*ent_to_right = *ent_to_left;
			}
		}
	}

	free(entries);
	return new_node;
}

static r_tree_node_t *choose_leaf_r(r_tree_t *rtree, const mbr_t *mbr)
{
	int mbr_enlarge, mbr_enlarge_min, mbr_area_min;
	int i, ient;
	r_tree_node_t *node = rtree->root;

	while(node->node_type != NODE_LEAF) {
		// get the entry whose rectangle needs least enlargement to include
		// the new polygon; resolve ties by choosing the entry with the
		// rectangle of smallest area
		mbr_enlarge_min = A_VERY_LARGE_NUM;
		for(i = 0; i < node->nr_entries; i++) {
			mbr_enlarge = mbr_area_inc(&node->entries[i].mbr, mbr);
			if(mbr_enlarge < mbr_enlarge_min) {
				mbr_area_min = mbr_area(&node->entries[i].mbr);
				mbr_enlarge_min = mbr_enlarge;
				ient = i;
			}
			else if(mbr_enlarge == mbr_enlarge_min) {
				if(mbr_area(&node->entries[i].mbr) < mbr_area_min) {
					mbr_area_min = mbr_area(&node->entries[i].mbr);
					ient = i;
				}
			}
		}
		node = node->entries[ient].ref.child;
	}

	return node;
}

static inline void set_leaf_entry(index_entry_t *entry, const mbr_t *mbr, const int idx)
{
	entry->mbr = *mbr;
	entry->ref.idx = idx;
}

static inline void set_nonleaf_entry(
	index_entry_t *entry,
	const mbr_t *mbr,
	const r_tree_node_t *child)
{
	entry->mbr = *mbr;
	entry->ref.child = child;
}

static void insert_r(r_tree_t *rtree, const mbr_t *mbr, int idx)
{
	r_tree_node_t *leaf, *new_leaf = NULL;

	// find position for new record
	leaf = choose_leaf_r(rtree, mbr);

	// if this leaf node has enough free space, insert the new record into it
	if(leaf->nr_entries < MAX_NR_ENTRIES_PER_NODE) {
		set_leaf_entry(&leaf->entries[(leaf->nr_entries)++], mbr, idx);
	}
	// otherwise, split the node
	else {
		new_leaf = split_node_r(rtree, leaf, mbr, (void *)&idx);
	}

	// adjust the tree, also passing the new leaf if a split happened
	adjust_tree_r(rtree, leaf, mbr, new_leaf);
}

int build_spatial_index_r(
	r_tree_t **index,
	const mbr_t *mbrs,
	const int nr_polys)
{
	int nr_nodes, idx;
	r_tree_t *rtree;

	rtree = malloc(sizeof(r_tree_t));
	if(!rtree) {
		return -1;
	}

	// we allocate an array of r-tree nodes, which enough
	// to accommodate the mbrs to be inserted
	nr_nodes = (nr_polys / (MIN_NR_ENTRIES_PER_NODE - 1) + 2);
	rtree->nodes = malloc(nr_nodes * sizeof(r_tree_node_t));
	if(!rtree->nodes) {
		free(rtree);
		return -1;
	}

	// initialize the root node
	rtree->root = &rtree->nodes[0];
	rtree->next_free = &rtree->nodes[1];
	rtree->max_nr_nodes = nr_nodes;
	rtree->nodes[0].node_type = NODE_LEAF;
	rtree->nodes[0].nr_entries = 0;
	rtree->nodes[0].parent_node = NULL;
	rtree->nodes[0].parent_entry = NULL;

	// insert all mbrs into the tree
	for(idx = 0; idx < nr_polys; idx++) {
		insert_r(rtree, &mbrs[idx], idx);
	}

	*index = rtree;
	return 0;
}

void free_spatial_index_r(r_tree_t *rtree)
{
	free(rtree->nodes);
	free(rtree);
}

// filtering based on the indexes built on both poly arrays
poly_pair_array_t *spatial_filter_r(
	const r_tree_t *rtree1,
	const r_tree_t *rtree2)
{
	poly_pair_block_t *first_block = NULL, *pblock;
	poly_pair_array_t *poly_pairs = NULL;
	int nr_poly_pairs = 0;

	// do filtering
	if(spatial_filter_r_1(rtree1->root, rtree2->root,
		&first_block, &nr_poly_pairs))
		goto error;

	// allocate a poly pair array to assemble separate
	// poly pair blocks together
	poly_pairs = malloc(sizeof(poly_pair_array_t));
	if(!poly_pairs)
		goto error;
	init_poly_pair_array(poly_pairs);

	// initialize poly pair array
	// similar to poly_array_t, we allocate a large continuous memory space
	// to accomodate mbrs, idx1 and idx2 together
	int size_mbrs = nr_poly_pairs * sizeof(mbr_t);
	int size_idx1 = nr_poly_pairs * sizeof(int);
	int size_idx2 = nr_poly_pairs * sizeof(int);

	poly_pairs->nr_poly_pairs = nr_poly_pairs;
	poly_pairs->mbrs = malloc(size_mbrs + size_idx1 + size_idx2);
	if(!poly_pairs->mbrs)
		goto error;
	poly_pairs->idx1 = (int *)((char *)(poly_pairs->mbrs) + size_mbrs);
	poly_pairs->idx2 = (int *)((char *)(poly_pairs->idx1) + size_idx1);

	// assemble separate poly pair blocks together into
	// a continuous poly pair array
	pblock = first_block;
	int i_poly_pair = 0;
	while(pblock) {
		memcpy(
			&poly_pairs->mbrs[i_poly_pair],
			pblock->mbrs,
			pblock->nr_poly_pairs * sizeof(mbr_t)
		);
		memcpy(
			&poly_pairs->idx1[i_poly_pair],
			pblock->idx1,
			pblock->nr_poly_pairs * sizeof(int)
		);
		memcpy(
			&poly_pairs->idx2[i_poly_pair],
			pblock->idx2,
			pblock->nr_poly_pairs * sizeof(int)
		);
		i_poly_pair += pblock->nr_poly_pairs;
		pblock = pblock->next_block;
	}

	goto success;

error:
	if(poly_pairs) {
		free_poly_pair_array(poly_pairs);
		poly_pairs = NULL;
	}

success:
	// free intermediate poly pair blocks
	while(first_block) {
		pblock = first_block->next_block;
		free(first_block);
		first_block = pblock;
	}

	return poly_pairs;
}

static int spatial_filter_r_1(
	const r_tree_node_t *node1,
	const r_tree_node_t *node2,
	poly_pair_block_t **first_block,
	int *nr_poly_pairs)
{
	index_entry_t *ent1, *ent2;
	int i1, i2, branch;

	branch =	(node1->node_type == NODE_LEAF) * 2 +
				(node2->node_type == NODE_LEAF);

	// we are filtering two non-leaves
	if(branch == 0) {
		for(i1 = 0; i1 < node1->nr_entries; i1++) {
			ent1 = &node1->entries[i1];
			for(i2 = 0; i2 < node2->nr_entries; i2++) {
				ent2 = &node2->entries[i2];
				if(mbr_intersect(&ent1->mbr, &ent2->mbr)) {
					if(spatial_filter_r_1(
						ent1->ref.child,
						ent2->ref.child,
						first_block,
						nr_poly_pairs
						)) {
						return -1;
					}
				}
			}
		}
	}

	// we are filtering a nonleaf and a leaf
	else if(branch == 1) {
		for(i1 = 0; i1 < node1->nr_entries; i1++) {
			ent1 = &node1->entries[i1];
			for(i2 = 0; i2 < node2->nr_entries; i2++) {
				ent2 = &node2->entries[i2];
				if(mbr_intersect(&ent1->mbr, &ent2->mbr)) {
					if(spatial_filter_r_2(
						ent1->ref.child,
						ent2,
						first_block,
						nr_poly_pairs
						)) {
						return -1;
					}
				}
			}
		}
	}

	// we are filtering a leaf and a nonleaf
	else if(branch == 2) {
		for(i1 = 0; i1 < node1->nr_entries; i1++) {
			ent1 = &node1->entries[i1];
			for(i2 = 0; i2 < node2->nr_entries; i2++) {
				ent2 = &node2->entries[i2];
				if(mbr_intersect(&ent1->mbr, &ent2->mbr)) {
					if(spatial_filter_r_3(
						ent1,
						ent2->ref.child,
						first_block,
						nr_poly_pairs
						)) {
						return -1;
					}
				}
			}
		}
	}

	// we are filtering two leaves
	else {
		for(i1 = 0; i1 < node1->nr_entries; i1++) {
			ent1 = &node1->entries[i1];
			for(i2 = 0; i2 < node2->nr_entries; i2++) {
				ent2 = &node2->entries[i2];
				if(mbr_intersect(&ent1->mbr, &ent2->mbr)) {
					if(filter_callback_r(
						&ent1->mbr, ent1->ref.idx,
						&ent2->mbr, ent2->ref.idx,
						first_block,
						nr_poly_pairs
						))
					{
						return -1;
					}
				}
			}
		}
	}

	return 0;
}

static int spatial_filter_r_2(
	const r_tree_node_t *node1,
	const index_entry_t *leaf_entry,
	poly_pair_block_t **first_block,
	int *nr_poly_pairs)
{
	index_entry_t *ent;
	int i;

	if(node1->node_type == NODE_LEAF) {
		for(i = 0; i < node1->nr_entries; i++) {
			ent = &node1->entries[i];
			if(mbr_intersect(&ent->mbr, &leaf_entry->mbr)) {
				if(filter_callback_r(
					&ent->mbr, ent->ref.idx,
					&leaf_entry->mbr, leaf_entry->ref.idx,
					first_block,
					nr_poly_pairs
					)) {
					return -1;
				}
			}
		}
	}
	else {
		for(i = 0; i < node1->nr_entries; i++) {
			ent = &node1->entries[i];
			if(mbr_intersect(&ent->mbr, &leaf_entry->mbr)) {
				if(spatial_filter_r_2(
					ent->ref.child,
					leaf_entry,
					first_block,
					nr_poly_pairs
					)) {
					return -1;
				}
			}
		}
	}

	return 0;
}

static int spatial_filter_r_3(
	const index_entry_t *leaf_entry,
	const r_tree_node_t *node2,
	poly_pair_block_t **first_block,
	int *nr_poly_pairs)
{
	index_entry_t *ent;
	int i;

	if(node2->node_type == NODE_LEAF) {
		for(i = 0; i < node2->nr_entries; i++) {
			ent = &node2->entries[i];
			if(mbr_intersect(&leaf_entry->mbr, &ent->mbr)) {
				if(filter_callback_r(
					&leaf_entry->mbr, leaf_entry->ref.idx,
					&ent->mbr, ent->ref.idx,
					first_block,
					nr_poly_pairs
					)) {
					return -1;
				}
			}
		}
	}
	else {
		for(i = 0; i < node2->nr_entries; i++) {
			ent = &node2->entries[i];
			if(mbr_intersect(&leaf_entry->mbr, &ent->mbr)) {
				if(spatial_filter_r_3(
					leaf_entry,
					ent->ref.child,
					first_block,
					nr_poly_pairs)) {
					return -1;
				}
			}
		}
	}

	return 0;
}

// return value: 0 - filtering continues; 1 - filtering stops
static inline int filter_callback_r(
	const mbr_t *mbr1, const int idx1,
	const mbr_t *mbr2, const int idx2,
	poly_pair_block_t **first_block,
	int *nr_poly_pairs)
{
	poly_pair_block_t *pblock;
	int i;

	// if we need to allocate a new poly pair block
	if(!(*nr_poly_pairs % POLY_PAIR_BLOCK_SIZE)) {
		pblock = malloc(sizeof(poly_pair_block_t));
		if(!pblock)
			return -1;
		pblock->nr_poly_pairs = 0;
		pblock->next_block = *first_block;
		*first_block = pblock;
	}

	// record this possibly intersecting polygong pair
	pblock = *first_block;
	i = pblock->nr_poly_pairs;
	mbr_merge(&pblock->mbrs[i], mbr1, mbr2);
	pblock->idx1[i] = idx1;
	pblock->idx2[i] = idx2;
	pblock->nr_poly_pairs = i + 1;

	(*nr_poly_pairs)++;

	return 0;
}
