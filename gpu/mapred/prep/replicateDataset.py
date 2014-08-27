#! /usr/bin/python

import sys
from collections import defaultdict

def main():
    # print getOffSets(sys.argv[1].strip(),8192,8192)
    # print getParentTile(sys.argv[1].strip())
    # return
    if len(sys.argv) < 2:
        sys.stderr.write("Usage: "+ sys.argv[0] + "\n")
        exit(1)

    repcount  = int(sys.argv[1])
    dic  = defaultdict(list)
    tab  = "\t"
    for line in sys.stdin:
        sp = line.strip().split(tab)
        assert len(sp) == 2
        tileId = int(sp[0])
        geom = sp[1]
        dic[tileId].append(geom)
    
    for i in xrange(repcount):
        offset = 6801 * i
        for tid,geom_tile in dic.items():
            tileId = tid + offset
            for polygon in geom_tile:
                print str(tileId)+ tab + polygon

    sys.stdout.flush()

if __name__ == '__main__':
    main()

