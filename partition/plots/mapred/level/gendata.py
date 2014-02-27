#! /usr/bin/python

import sys
import math
from collections import defaultdict
import numpy as np

def main():

    dicc = defaultdict(list)
    for line in sys.stdin:
        sp = line.strip().split(",")
        dicc[sp[0]].append(float(sp[1]))

    # for key,val in dicc.items():
    data =[]
    for key in ("slc","bos","str"):
        l = dicc[key]
        avg_val = np.average(l)
        deviation  = np.std(l)
        data.append(avg_val)
        data.append(deviation)


    print " ".join(str(x) for x in data)

    sys.stdout.flush()
    sys.stderr.flush()

if __name__ == '__main__':
    main()

