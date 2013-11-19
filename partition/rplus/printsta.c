#include <stdio.h>
#include "assert.h"
#include "macros.h"
#include "options.h"
#include "index.h"
#include "global.h"

int     NO_REFIN;

InsertStats()
{
        printf("\n");
        printf("rectangles inserted:\t\t%d\n", InsertCount);
        printf("splits:\t\t\t\t%d\n", InSplitCount);
        if (InSplitCount > 0)
                printf("average merit of splits:\t%3.2f\n",
                SplitMeritSum / InSplitCount);
        if (InsertCount > 0)
                printf("splits per insert:\t\t%3.2f\n",
                (float)InSplitCount/InsertCount);

    /*  printf("partition evaluations:\t\t%d\n", EvalCount);
    **  if (InSplitCount > 0)
    **  {
    **          printf("evaluations per split:\t\t%3.2f\n",
    **          (float)EvalCount / InSplitCount);
    **  }
    */

        if (CallCount > 0 && InSplitCount > 0)
        {
                printf("recursive calls per split:\t%d\n",
                CallCount / InSplitCount);
        }

    /*  if (InsertCount > 0)
    **  {
    **          printf("evaluations per insert:\t\t%3.2f\n",
    **          (float)EvalCount / InsertCount);
    **  }
    */

        if (InsertCount > 0)
        {
                printf("pages touched per insert:\t%3.2f\n",
                (float)InTouchCount / InsertCount);
        }

        printf("# of downward splits:\t\t%d\n", DownwardSplitCount);
        if (InSplitCount > 0)
                printf("average downward splits per split:\t%3.2f\n",
                (float) DownwardSplitCount / InSplitCount);
        if (InsertCount > 0)
                printf("average downward splits per insert:\t%3.2f\n",
                (float)DownwardSplitCount / InsertCount);

        printf("cpu time:\t\t\t%f\n", UserTime + SystemTime);
        if (InsertCount > 0)
                printf("cpu time per insert:\t\t%f\n",
                        (UserTime+SystemTime) / InsertCount);
        if (InSplitCount > 0)
                printf("cpu time per split:\t\t%f\n",
                        (UserTime+SystemTime) / InSplitCount);
        TreeStats();
}

SearchStats()
{
        float fraction;

        printf("\n");
        printf("searches:\t\t\t\t%d\n", SearchCount);
        printf("qualifying rects:\t\t\t%d\n", HitCount);
	printf("qualifying rects (no refinement):\t%d\n", NO_REFIN);
        if (SearchCount > 0)
                printf("avg hits per search:\t\t\t%3.2f\n", (float)HitCount / SearchCount);
        if (RectCount*SearchCount > 0)
        {
                fraction = (float)HitCount / (RectCount * SearchCount);
                printf("avg fraction of data each search:\t%d%%\n",
                        (int)(100*fraction+0.5));
        }
        if (SearchCount > 0)
                printf("pages touched per search:\t\t%3.2f\n",
                (float)SeTouchCount / SearchCount);
        if (HitCount > 0)
        {
                printf("pages touched per hit:\t\t\t%3.2f\n",
                (float)SeTouchCount / HitCount);
        }
        printf("cpu time:\t\t\t\t%f\n", UserTime + SystemTime);
        if (SearchCount > 0)
                printf("cpu time per search:\t\t\t%f\n",
                        (UserTime+SystemTime) / SearchCount);
        if (HitCount > 0)
                printf("cpu time per hit:\t\t\t%f\n", (UserTime+SystemTime) / HitCount);
}

void DeleteStats()
{
        printf("\n");
        printf("rectangles deleted:\t\t%d\n", DeleteCount);

    /*  printf("nodes eliminated:\t\t%d\n", ElimCount);
    **  if (DeleteCount > 0)
    **          printf("nodes eliminated per delete:\t%3.3f\n",
    **                  (float)ElimCount / DeleteCount);
    **  printf("nodes reinserted:\t\t%d\n", ReInsertCount);
    **  if (ElimCount > 0)
    **          printf("reinsertions per node elim:\t%3.2f\n",
    **          (float)ReInsertCount / ElimCount);
    **  if (DeleteCount > 0)
    **          printf("reinsertions per rect delete:\t%3.3f\n",
    **          (float)ReInsertCount / DeleteCount);
    **  printf("node splits from reinsertions:\t%d\n", DeSplitCount);
    */

        if (DeleteCount > 0)
        {
                printf("pages touched per delete:\t%3.2f\n",
                (float)DeTouchCount / DeleteCount);
        }
        printf("cpu time:\t\t\t%f\n", UserTime + SystemTime);
        if (DeleteCount > 0)
                printf("cpu time per delete:\t\t%f\n",
                        (UserTime+SystemTime) / DeleteCount);
}

void GeneralStats()
{

        printf("\n");
        printf("bytes per page:\t\t\t%d\n", PAGESIZE);
        printf("dimensions:\t\t\t%d\n", NUMDIMS);
        printf("entries per node:\t\t%d\n", NODECARD);
}

void TreeStats()
{
        assert(RectCount >= 0);

        printf("\n");
        printf("rectangles indexed:\t\t%d\n", RectCount);
        printf("levels in tree:\t\t\t%d\n", Root->level + 1);
        printf("nodes in tree:\t\t\t%d\n", NodeCount);
        printf("leaf nodes:\t\t\t%d\n", LeafCount);
        printf("non-leaf nodes:\t\t\t%d\n", NonLeafCount);
        if (RectCount > 0)
                printf("index nodes per data item:\t%3.2f\n",
                        (float)NodeCount / RectCount);
        printf("size of index in bytes:\t\t%d\n", NodeCount * PAGESIZE);
        if (RectCount > 0)
                printf("index bytes per data item:\t%3.2f\n",
                        (float)(NodeCount * PAGESIZE) / RectCount);
        printf("total entries:\t\t\t%d\n", EntryCount);
        printf("total slots for entries:\t%d\n", NodeCount*NODECARD);
        if (NodeCount > 0)
                printf("space utilization:\t\t%d%%\n",
                        (int)(100*(float)EntryCount/(NodeCount*NODECARD)));
}

