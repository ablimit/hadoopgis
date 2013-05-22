#! /home/aaji/softs/bin/python

import sys
import math
#from collections import defaultdict
#dic = defaultdict(list)

dic = {}

def main():

    if len(sys.argv) < 2:
        sys.stderr.write("Usage: "+ sys.argv[0] + " [oid \t pid]\n")

    for line in open(sys.argv[1],'r'):
        sp =line.split("\t")
        dic [int(sp[0])] = int(sp[1])

    for line in sys.stdin:
        sp =line.strip().split("|")
        if len(sp)>1:
            oid = int(sp[1])
            print '|'.join((sp[0],str(oid),str(dic[oid]),sp[2]))

    sys.stdout.flush()


if __name__ == '__main__':
    main()
