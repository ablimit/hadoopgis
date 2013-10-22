#define Undefined(x) ((x)->boundary[0] > (x)->boundary[NUMDIMS])

struct OverFlowNode * GetOverFlowPage (int idxp, off_t offset);
struct Rect CombineRect(struct Rect *r, struct Rect *rr);
struct Rect IntersectRect(struct Rect *r, struct Rect *rr);
