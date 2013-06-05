#! /home/aaji/softs/bin/python

import sys
import re
import math
from collections import defaultdict

dic = defaultdict(list)


# Calculate mean of the values
def normx(x,osm):
    if osm:
        return (x + 180.0)/360.0 
    else:
        return x/110592.0

def normy(y,osm):
    if osm:
        return (y + 90.0)/180.0
    else:
        return y/57344.0

def main():
    if len(sys.argv) <2:
        print "Usage: "+ sys.argv[0] + " [osm| pais]"
        sys.exit(1)

    osm = False
    if "osm" in sys.argv[1].lower():
        osm = True


    for line in sys.stdin:
        sp = line.strip().split()
        idx = 0
        if (len(sp)==6):
            idx = 1
        x1 = normx(float(sp[1+idx]),osm)
        y1 = normy(float(sp[2+idx]),osm)
        x2 = normx(float(sp[3+idx]),osm)
        y2 = normy(float(sp[4+idx]),osm)
	    # print "after norm: %f" % x1
        if idx ==0:
            print "\t".join((sp[0],str(x1),str(y1),str(x2),str(y2)))
        else:
            print "\t".join((sp[0],sp[1],str(x1),str(y1),str(x2),str(y2)))

        #else:
        #print len(sp)
    sys.stdout.flush()

if __name__ == '__main__':
    main()

