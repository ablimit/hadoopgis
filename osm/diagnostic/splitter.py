#! /usr/bin/python

import sys
from collections import defaultdict

dic = defaultdict(list)

for line in sys.stdin:
    sp =line.split("|",4)
    dic[sp[1]].append(line)

for key,val in dic.items():
    f = open(key,'w')
    for line in val:
	f.write(line)
    f.close()

