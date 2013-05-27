#! /home/aaji/softs/bin/python

import sys
import re
import math
from collections import defaultdict

dic = {}

pid_oid = defaultdict(list)

# Calculate standard deviation
def stddev(values, mean):
    size = len(values)
    sum = 0.0
    for n in range(0, size):
	sum += math.sqrt((values[n] - mean)**2)
    return math.sqrt((1.0/(size-1))*(sum/size))


# Calculate mean of the values
def intersects(a,b):
    return not(a[2] <= b[0] or b[2]<=a[0] or a[1] >= b[3] or b[1] >= a[3])


def update_partition(oid,object_mbb):
    for pid, partition_mbb in dic.items():
	if intersects(object_mbb,partition_mbb):
	    #print "\t".join((oid,pid))
	    pid_oid[pid].append(oid)

def main():
    if len(sys.argv) <2:
        print "Usage: "+ sys.argv[0] + " [partition info]"
        sys.exit(1)


    for line in open(sys.argv[1],'r'):
        sp = line.strip().split()
        if (len(sp)>4):
            x1 = float(sp[1])
            y1 = float(sp[2])
            x2 = float(sp[3])
            y2 = float(sp[4])
            dic[sp[0]]=(x1,y1,x2,y2)

    # sys.stderr.write("Partition Size: " + str(len(dic))+"\n")

    for line in sys.stdin:
        sp = line.strip().split()
        if (len(sp)>4):
            x1 = float(sp[1])
            y1 = float(sp[2])
            x2 = float(sp[3])
            y2 = float(sp[4])
            update_partition(sp[0],(x1,y1,x2,y2))
	    #print "\t".join((sp[0],str(x1),str(y1),str(x2),str(y2)))
        #else:
        #print len(sp)
    
    cardins =[]
    for pid, oid_list in pid_oid.items():
	cardins.append(len(oid_list))
	print "\t".join((pid,"\t".join(oid_list)))

    cardins.sort()
    min_val = min(cardins)
    max_val = max(cardins)
    avg_val = int(float(sum(cardins))/float(len(cardins)))
    median = cardins[len(cardins)/2]
    deviation  = stddev(cardins,avg_val)

    sys.stderr.write("\t".join((str(min_val),str(max_val),str(avg_val),str(median),str(deviation),"\n")))

    sys.stdout.flush()
    sys.stderr.flush()

if __name__ == '__main__':
    main()


