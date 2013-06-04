#! /usr/bin/python

import sys
import re
import math
from collections import defaultdict

dic = {}

pid_oid = defaultdict(list)

# Calculate standard deviation
def stddev(values, mean):
    size = len(values)
    sum = 0.0
    for n in range(0, size):
	sum += math.sqrt((values[n] - mean)**2)
    return math.sqrt((1.0/(size-1))*(sum/size))


# Calculate mean of the values
def intersects(a,b):
    return not(a[2] <= b[0] or b[2]<=a[0] or a[1] >= b[3] or b[1] >= a[3])

def intersection(a,b):
    width  = abs((a[2]- b[0]) if a[2] >= b[0] >= a[0] else (b[2]-a[0]))
    height = abs((a[3]- b[1]) if a[3] >= b[1] >= a[1] else (b[3]-a[1]))
    area = width * height
    return area


def enlarge(a,b):
    x = a[0] if a[0] < b[0] else b[0]
    y = a[1] if a[1] < b[1] else b[1]
    xx = a[2] if a[2] > b[2] else b[2]
    yy = a[3] if a[3] > b[3] else b[3]
    return (x,y,xx,yy)

def getOverlap(boxes):
    overlap =0.0
    for i in range(0,len(boxes)-1):
        src = boxes [i]
        for j in range(i+1,len(boxes)):
            tar = boxes [j]
            if intersects(src,tar):
                overlap += intersection(src,tar) 
    return overlap

def area(rect):
    return (rect[3] - rect [1] ) * (rect[2] - rect[0])

def getBox(boxes):
    mbb = boxes[0]
    for box in boxes[1:]:
        mbb = enlarge(mbb,box)
    return mbb

def eval_partition(pinfo):
    box = area(getBox(pinfo.values()))
    overlap = getOverlap(pinfo.values())
    return overlap/box


def main():
    global dic
    global pid_oid
    
    cardins =[]
    
    for line in sys.stdin:
        sp = line.strip().split()
        if (len(sp)>5):
            x1 = float(sp[1])
            y1 = float(sp[2])
            x2 = float(sp[3])
            y2 = float(sp[4])
            cardin.append(int(sp[5]))
            dic[int(sp[0])]=(x1,y1,x2,y2)

    # sys.stderr.write("Partition Size: " + str(len(dic))+"\n")

    cardins.sort()
    min_val = min(cardins)
    max_val = max(cardins)
    avg_val = int(float(sum(cardins))/float(len(cardins)))
    median = cardins[len(cardins)/2]
    deviation  = stddev(cardins,avg_val)
    # ratio of overlap/MBB
    r = eval_partition(dic)
    
    print "\t".join(str(x) for x in (min_val,max_val,avg_val,median,deviation,r))

    sys.stdout.flush()
    sys.stderr.flush()

if __name__ == '__main__':
    main()


