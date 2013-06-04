#! /usr/bin/python

import sys
import math

def main():
    if len(sys.argv) <2:
        print "Usage: "+ sys.argv[0] + " [output image path]"
        sys.exit(1)
    path = sys.argv[1].strip()

    print "set terminal pngcairo size 1024,768 enhanced font 'Verdana,20'"
    print "set output '{0}'".format(path)
    print "unset xtics" 
    print "unset ytics"

    oid = 0
    for line in sys.stdin:
        sp = line.strip().split()
        if (len(sp)>4):
            oid += 1
            x1 = sp[1]
            y1 = sp[2]
            x2 = sp[3]
            y2 = sp[4]
            print "set object {0} rect from {1}, {2} to {3}, {4}".format(oid,x1,y1,x2,y2)
    
    print "plot [-0.05:1.05] [-0.05:1.05] NaN notitle "

    sys.stdout.flush()
    sys.stderr.flush()

if __name__ == '__main__':
    main()

