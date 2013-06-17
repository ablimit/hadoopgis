#! /home/aaji/softs/bin/python

import sys
import math
from collections import defaultdict

dic = defaultdict(dict)


def main():
    comma = ","
    for line in sys.stdin:
        sp =line.strip().split(comma)
        if len(sp) == 4:
            dic[int(sp[2])][sp[0]]=sp[3]
    
    print comma.join(("reducer","oc2500","oc5000","oc10000","oc15000","oc20000","oc25000","oc30000","oc50000"))
    for key in sorted(dic.iterkeys()):
        val = dic[key]
        print comma.join((str(key),val["oc2500"],val["oc5000"],val["oc10000"],val["oc15000"],val["oc20000"],val["oc25000"],val["oc30000"],val["oc50000"]))

if __name__ == '__main__':
    main()


