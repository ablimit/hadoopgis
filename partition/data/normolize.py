#! /home/aaji/softs/bin/python

import sys
import re
import math
from collections import defaultdict

dic = defaultdict(list)

osm = False

# Calculate mean of the values
def normx(x):
    if osm:
        return (x + 180.0)/360.0 
    else:
        return x/110592.0

def normy(y):
    if osm:
        return (y + 90.0)/180.0
    else:
        return y/57344.0

def main():
    if len(sys.argv) <2:
        print "Usage: "+ sys.argv[0] + " [osm| pais]"
        sys.exit(1)

    if "osm" in sys.argv[1].lower():
        osm = True
    else:
        osm = False

    for line in sys.stdin:
        sp = line.strip().split()
        if (len(sp)>4):
            x1 = normx(float(sp[1]))
            y1 = normy(float(sp[2]))
            x2 = normx(float(sp[3]))
            y2 = normy(float(sp[4]))
            print "\t".join((sp[0],str(x1),str(y1),str(x2),str(y2)))
        #else:
        #print len(sp)
    sys.stdout.flush()

if __name__ == '__main__':
    main()


