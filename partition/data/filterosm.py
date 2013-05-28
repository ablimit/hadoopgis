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
        if (len(sp)>4):
            x1 = float(sp[1])
            y1 = float(sp[2])
            x2 = float(sp[3])
            y2 = float(sp[4])
            area = (x2-x1)*(y2-y1)
            items.append((sp[0],x1,y1,x2,y2,area))
        #else:
        #print len(sp)

    # sort objects by area 
    items.sort(key=lambda x: x[5], reverse=True)
    
    #omit first million items 
    for item in items[1000000:]:
        print "\t".join(str(x) for x in item)

    sys.stdout.flush()

if __name__ == '__main__':
    main()


