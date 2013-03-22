#! /home/aaji/softs/bin/python

import sys
import math
from collections import defaultdict

dic = defaultdict(list)


# Calculate mean of the values
def mean(values):
    size = len(values)
    sum = 0.0
    for n in range(0, size):
	sum += values[n]
	return sum / size;

# Calculate standard deviation
def stddev(values, mean):
    size = len(values)
    sum = 0.0
    for n in range(0, size):
	sum += math.sqrt((values[n] - mean)**2)
    return math.sqrt((1.0/(size-1))*(sum/size))


for line in sys.stdin:
    sp =line.split(",")
    if len(sp)>1:
	dic[sp[0] if len(sp)==2 else '|'.join((sp[0],sp[1]))].append(int(sp[-1])-5)


for key,val in dic.items():
    mu = mean(val)
    dev = stddev(val,mu)
    print key,mu,dev

