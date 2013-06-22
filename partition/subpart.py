#! /home/aaji/softs/bin/python

import sys
import re
import math
import gzip
from collections import defaultdict


def main():
    if len(sys.argv) <3:
        print "Usage: "+ sys.argv[0] + " [zipped mbb file] [output directory]"
        sys.exit(1)
    path = sys.argv[1]
    outdir= sys.argv[2]
    if outdir[-1] == '/':
	outdir = outdir[0:-1]

    if path[-1] != '.gz':
	sys.stderr.write(path +" is not a gzipped file.\n")
	exit(1)

    output_dic = defaultdict(list)
    
    dic = {}
    ID_IDX = 0
    TAG_IDX =1
    POLYGON_IDX = -1
    
    for line in sys.stdin:
        sp = line.strip().split()
        if (len(sp) >= 2):
            for i in xrange(1,len(sp)):
                dic[sp[i]] = sp[0]

    cc =0 
    for line in gzip.open(path,'r'):
	sp = line.strip().split()
	oid = "-".join((sp[ID_IDX],sp[TAG_IDX]))
	if oid in dic:
	    output_dic[dic[oid]].append(line.strip())
	else:
	    cc +=1

    for key,val in output_dic.items():
	f = open(outdir+"/"+key, 'w')
	for line in val:
	    f.write('%r\n' % line)
	f.close()

    sys.stderr.write("["+str(cc)+"] objects is missing.\n")
    sys.stdout.flush()
    sys.stderr.flush()

if __name__ == '__main__':
    main()

