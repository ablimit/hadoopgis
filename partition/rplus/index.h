/*-----------------------------------------------------------------------------
| Global definitions.
-----------------------------------------------------------------------------*/

struct Rect
{
        int boundary[NUMSIDES];
};

struct Branch
{
        struct Rect minrect;  /* minimal coverage - seeps up Search */
        struct Rect rect;
        int son;
};

struct LBranch
{
        struct Rect rect;
        int son;
};

struct Node
{
        short int count;
        short int level; /* 0 is leaf, others positive */
	int pageNo;
        struct Rect rect;  /* duplicate of rect in parent branch that describes
                              this node - used instead of a parent pointer */
        struct Branch branch[NODECARD];
};

struct OverFlowNode {
  	short int count;
	int pageNo;
	struct OverFlowNode *next2;
	struct Rect rect[OVERFLOWNODECARD];
};


struct ListNode
{
        struct ListNode *next;
        struct Node *node;
};

/* rp: list of rectangles when packing the tree */
struct ListRect
{
        struct ListRect *next;
        struct Rect *rect;
};

struct NodeCut
{
        char axis;  /* 'x' for vertical cut; 'y' for horizontal cut */
        int cut;
};
