#! /usr/bin/python

import sys
import math
from collections import defaultdict

def main():
    if len(sys.argv) <2:
        print "Usage: "+ sys.argv[0] + " [field]"
        sys.exit(1)

    field = sys.argv[1].strip()
    idx = 0
    if field == "min":
        idx = 2
    elif field == "max":
        idx = 3
    elif field == "avg":
        idx = 4
    elif field == "median":
        idx = 5
    elif field == "count":
        idx = 6
    elif field =="stddev":
        idx = 7
    elif field == "ratio":
        idx = 8
    else:
        sys.stdout.write("unrecognized field name: " + field + "\n")
        sys.exit(1)

    keys = [864,4322,8644,17288,43220,86441,172882,432206,864412,4322062]

    labels = ["0001", "0005", "001", "0015", "0020", "0025", "01", "05", "10",
              "15", "20", "25", "100"] 
    datalog = defaultdict(dict) 
    for line in sys.stdin:
        sp = line.strip().split(",")
        if (len(sp) >= 2):
            # st + x + y 
            #key = int(sp[2]) if sp[0]=="st" else int(sp[1])
            #method_name = "-".join((sp[0],sp[1])) if sp[0] == "st" else sp[0]
            #val  = sp[idx+1] if sp[0] == "st" else sp[idx]

            key = int(sp[1])
            method_name = sp[0]
            val  = sp[idx]

            datalog[key][method_name]= val

    printheader = True
    for x in keys:
        if printheader:
            print "x " +" ".join(["1.0" if l == "100" else "0."+l for l in
                                  labels])
            printheader = False
        vals = [datalog[x][name] if name in datalog[x] else "?" for name in labels]
        vals.insert(0,str(x))
        print " ".join(vals)

    sys.stdout.flush()
    sys.stderr.flush()

if __name__ == '__main__':
    main()

