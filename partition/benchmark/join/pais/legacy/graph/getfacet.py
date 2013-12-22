#! /home/aaji/softs/bin/python

import sys
import math
from collections import defaultdict

dic = defaultdict(dict)


def main():
    if len(sys.argv) <2:
        print "Usage: "+ sys.argv[0] + " [partition id]"
        sys.exit(1)
    
    pid = sys.argv[1].strip()
    # print >> sys.stderr, "Pid is", pid
    comma = ","
    for line in sys.stdin:
        sp =line.strip().split(comma)
        if len(sp) == 4 and sp[0].lower() == pid.lower():
            dic[int(sp[2])][sp[1]]=sp[3]
    
    print comma.join(("reducer","rtree","minskew","rkHist","rv"))
    for key in sorted(dic.iterkeys()):
        val = dic[key]
        print comma.join((str(key),val["rtree"],val["minskew"],val["rkHist"],val["rv"]))

if __name__ == '__main__':
    main()


