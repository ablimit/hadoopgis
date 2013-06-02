#! /home/aaji/softs/bin/python

import sys
import re
import math

def main():
    if len(sys.argv) <2:
        print "Usage: "+ sys.argv[0] + " [pais image]"
        sys.exit(1)

    dic = {}
    for line in sys.stdin:
        sp = line.strip().split()
        if (len(sp) >= 2):
            for i in xrange(1,len(sp)):
                dic[int(sp[i])] = int(sp[0])

    cc =0 ;
    for line in open(sys.argv[1],'r'):
        sp = line.strip().split("|")
        oid = int(sp[1])
        if oid in dic:
            pid = dic[oid]
            print "|".join((sp[0],str(pid),str(oid),sp[2]))
        else:
            cc +=1

    sys.stderr.write("["+str(cc)+"] objects is missing.\n")
    sys.stdout.flush()
    sys.stderr.flush()

if __name__ == '__main__':
    main()


