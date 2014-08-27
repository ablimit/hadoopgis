#! /usr/bin/python

import sys
from collections import defaultdict

def main():
    statDic = defaultdict(int)
    for line in sys.stdin:
        sp = line.strip().split("\t")
        assert len(sp) == 2
        tileId = int(sp[0])
        statDic[tileId] += 1 
    
    for tid,ocount in statDic.items():
        print "\t".join((str(tid),str(ocount)))
    sys.stdout.flush()

if __name__ == '__main__':
    main()

