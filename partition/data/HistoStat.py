#! /usr/bin/python

import sys
from collections import defaultdict
from collections import Counter

def main():
    olist = []
    for line in sys.stdin:
        k = line.strip()
        if len(k) == 0:
            k = "bound"
        olist.append(k)
    
    histo = Counter(olist)
    for k,c in histo.items():
        print "\t".join(k,str(c))

    sys.stdout.flush()

if __name__ == '__main__':
    main()

