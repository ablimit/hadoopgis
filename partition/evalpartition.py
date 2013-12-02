#! /usr/bin/python

import sys
import math
from collections import Counter
import numpy as np

def dataset_size(filename):
    i = 0
    with open(filename,'r') as f:
        for i in f:
            i += 1
    return i


def main():
    if len(sys.argv) <2:
        print "Usage: "+ sys.argv[0] + " [partition file]"
        sys.exit(1)

    dicc = Counter()
    raw_ds_size = sys.argv[1].strip()
    
    for line in sys.stdin:
        sp = line.strip().split()
        if (len(sp) == 6):
            dicc[int(sp[0])] +=1

    pinfo = dicc.values()
    min_val = np.amin(pinfo)
    max_val = np.amax(pinfo)
    avg_val = np.average(pinfo)
    median = np.median(pinfo)
    deviation  = np.std(pinfo)
    partition_count = len(pinfo)
    cooked_ds_size = sum(pinfo)
    ratio = float(cooked_ds_size) / float(raw_ds_size) - 1.0
    
    print ",".join(str(x) for x in [min_val,max_val,avg_val,median,partition_count,deviation,ratio])

    sys.stdout.flush()
    sys.stderr.flush()

if __name__ == '__main__':
    main()

