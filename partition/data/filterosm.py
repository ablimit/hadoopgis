#! /home/aaji/softs/bin/python

import sys
import re
import math
#from collections import defaultdict
#dic = defaultdict(list)

def main():
    items =[]
    for line in sys.stdin:
        sp = line.strip().split()
        idx = 0
        if (len(sp)==6):
            idx = 1
	x1 = float(sp[1+idx])
        y1 = float(sp[2+idx])
        x2 = float(sp[3+idx])
        y2 = float(sp[4+idx])
        area = (x2-x1)*(y2-y1)
        items.append((sp[0],x1,y1,x2,y2,area))
        #else:
        #print len(sp)

    # sort objects by area 
    items.sort(key=lambda x: x[5], reverse=True)
    
    #omit first million items 
    for item in items[18000000:]:
        print "\t".join(str(x) for x in item)

    sys.stdout.flush()

if __name__ == '__main__':
    main()


