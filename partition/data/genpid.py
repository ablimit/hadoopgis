#! /home/aaji/softs/bin/python

import sys
import re
import math
from collections import defaultdict

dic = {}

pid_oid = defaultdict(list)

# Calculate mean of the values
def intersects(a,b):
    return not(a[2] <= b[0] or b[2]<=a[0] or a[1] >= b[3] or b[1] >= a[3])

def update_partition(oid,object_mbb):
    global dic
    global pid_oid
    flag = False
    for pid, partition_mbb in dic.items():
        if intersects(object_mbb,partition_mbb):
            #print "\t".join((oid,pid))
            pid_oid[pid].append(oid)
            flag =True

    if not flag:
        sys.stderr.write("invalid oid [" + oid + "]\n")

def main():
    if len(sys.argv) <2:
        print "Usage: "+ sys.argv[0] + " [partition info]"
        sys.exit(1)

    global dic
    global pid_oid
    
    # readin partition info and store by id
    for line in open(sys.argv[1],'r'):
        sp = line.strip().split()
        if (len(sp)>4):
            x1 = float(sp[1])
            y1 = float(sp[2])
            x2 = float(sp[3])
            y2 = float(sp[4])
            dic[sp[0]]=(x1,y1,x2,y2)

    # sys.stderr.write("Partition Size: " + str(len(dic))+"\n")

    # readin object mbb data 
    for line in sys.stdin:
        sp = line.strip().split()
        idx = 0
        if (len(sp)==6):
            idx = 1
        x1 = float(sp[1+idx])
        y1 = float(sp[2+idx])
        x2 = float(sp[3+idx])
        y2 = float(sp[4+idx])
        oid = "-".join((sp[0],sp[1])) if idx ==1 else sp[0]
        update_partition(oid,(x1,y1,x2,y2))
	    #print "\t".join((sp[0],str(x1),str(y1),str(x2),str(y2)))
        #else:
        #print len(sp)
    
    for pid, oid_list in pid_oid.items():
        print "\t".join((pid,"\t".join(oid_list)))

    sys.stdout.flush()
    sys.stderr.flush()

if __name__ == '__main__':
    main()

