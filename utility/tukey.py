#! /home/aaji/softs/bin/python

import sys
import math


# Calculate mean of the values
def mean(values):
    return sum(values)/len(values)

# Calculate standard deviation
def stddev(values, mean):
    size = len(values)
    sum = 0.0
    for n in range(0, size):
	sum += math.sqrt((values[n] - mean)**2)
    return math.sqrt((1.0/(size-1))*(sum/size))


vals =[]
for line in sys.stdin:
    vals.append(int(line.strip()))


mu = mean(vals)
dev = stddev(vals,mu)
print min(vals),max(vals),vals[len(vals)//2],mu,dev

