#! /home/aaji/softs/bin/python

import sys
import math

def main():
    for line in sys.stdin:
        sp = line.strip().split()
        if (len(sp)>4):
	    oid = sp[0]
            x1 = sp[1]
            y1 = sp[2]
            x2 = sp[3]
            y2 = sp[4]
	    print "set object ${oid} rect from ${x}, ${y} to ${xx}, ${yy}" >> ${tempf}
    
            cat partres/${f}.${method}.txt | while read oid x y xx yy a;
    sys.stdout.flush()
    sys.stderr.flush()

if __name__ == '__main__':
    main()


