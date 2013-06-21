#! /home/aaji/softs/bin/python

import sys
import math
from collections import defaultdict

dic = defaultdict(list)


def main():
    # print >> sys.stderr, "Pid is", pid
    comma = ","
    for line in sys.stdin:
        sp =line.strip().split(comma)
        if len(sp) == 3:
            dic[int(sp[1])].append(sp[2])
    
    # print comma.join(("reducer","rtree","minskew","rkHist","rv"))
    for key in sorted(dic.iterkeys()):
        val = dic[key]
        print comma.join((str(key),comma.join(val)))

if __name__ == '__main__':
    main()


