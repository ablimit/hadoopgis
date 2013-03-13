#! /usr/bin/python

import sys
from collections import defaultdict

dic = defaultdict(list)

for line in sys.stdin:
    sp =line.split(",")
    if len(sp)>2:
	dic['|'.join((sp[0],sp[1]))].append(int(sp[-1])-5)

for key,val in dic.items():
    print key, float(sum(val))/float(len(val))


