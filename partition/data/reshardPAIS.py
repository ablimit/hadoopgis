#! /home/aaji/softs/bin/python

import sys
import re
import math
from collections import defaultdict

def main():
    if len(sys.argv) <2:
        print "Usage: "+ sys.argv[0] + " [pais image]"
        sys.exit(1)

    dic = defaultdict(list)
    for line in sys.stdin:
        sp = line.strip().split()
	sys.stderr.write("line lenth: "+str(len(sp))+"\n")
        if (len(sp) >= 2):
            for i in xrange(1,len(sp)):
                dic[sp[i]].append(int(sp[0]))
		# if dic[sp[i]] ==1:
		#    sys.stderr.write("yay\n")
	else:
	    sys.stderr.write("Rediculous.\n")

    cc =0 ;
    for path in sys.argv[1:]:
        tag = path[-1]
	sys.stderr.write("tag:" + tag +"\n")
        for line in open(path,'r'):
            sp = line.strip().split("|")
            raw_oid =str(int(sp[1]))
            oid = "-".join((raw_oid,tag))
	    # sys.stderr.write("oid: "+oid+ "\n")
            if oid in dic:
		for pid in dic[oid]:
		    if pid == 1:
			sys.stderr.write("ohoo\n")
		    print "|".join((sp[0],tag,str(pid),raw_oid,sp[2]))
	    else:
		cc +=1

    sys.stderr.write("["+str(cc)+"] objects is missing.\n")
    sys.stdout.flush()
    sys.stderr.flush()

if __name__ == '__main__':
    main()

