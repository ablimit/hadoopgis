#! /usr/bin/python

import sys
from collections import defaultdict

dic = defaultdict(list)

for line in sys.stdin:
    sp =line.strip().split("\t",1)
    dic[sp[0]].append(sp[1])

for key,val in dic.items():
    print key + "\t" + "|".join(val)

