#! /home/aaji/softs/bin/python

import sys
import re
import math

def main():
    if len(sys.argv) <2:
        print "Usage: "+ sys.argv[0] + " [osm Map]"
        sys.exit(1)

    dic = {}
    ID_IDX = 0
    POLYGON_IDX = -1
    for line in sys.stdin:
        sp = line.strip().split()
        if (len(sp) >= 2):
            for i in xrange(1,len(sp)):
                dic[sp[i]] = sp[0]

    cc =0 ;
    for path in sys.argv[1:]:
        tag = path[-1]
        for line in open(path,'r'):
            sp = line.strip().split("|")
            raw_oid =str(int(sp[ID_IDX]))
            oid = "-".join((raw_oid,tag))
            if oid in dic:
                pid = dic[oid]
                print "|".join((tag,pid,raw_oid,sp[POLYGON_IDX]))
            else:
                cc +=1

    sys.stderr.write("["+str(cc)+"] objects is missing.\n")
    sys.stdout.flush()
    sys.stderr.flush()

if __name__ == '__main__':
    main()

