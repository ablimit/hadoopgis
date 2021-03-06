#! /usr/bin/python

import sys
from collections import defaultdict

def readITMap(path):
    image_tile_map = defaultdict(dict)
    tid = 1
    for line in open(path,'r'):
        sp =line.strip().split('|')
        image = sp[1]
        tile  = sp[0] 
        image_tile_map[image][tile] = tid
        tid += 1 
    return image_tile_map

def main():
    if len(sys.argv) < 2:
        sys.stderr.write("Usage: "+ sys.argv[0] + "\n")
        exit(1)

    path = sys.argv[1]
    dic  = readITMap(path)
    tab  = "\t"
    seq_oid = [0,0]
    for line in sys.stdin:
        sp = line.strip().split("|")
        assert len(sp) == 5
        tile = sp[0]
        ti = tile.split('-')
        xOffset = int(ti[0])
        yOffset = int(ti[1])

        image = sp[1] 
        did = int(sp[2])
        seq_oid[did-1] +=1
        oid = seq_oid [did-1]
        shape = sp[4][9:-2]
        sp = shape.split(',')
        nver = str(len(sp))
        r,t = 0,0
        l,b = sys.maxint,sys.maxint;
        shp = []
        for point in sp:
            coor = point.split()
            x = int(coor[0])
            y = int(coor[1])
            if x < l:
                l = x

            if x > r:
                r = x

            if y < b:
                b = y

            if y > t:
                t = y

            shp.append((str(x-xOffset), str(y-yOffset)))

        l = str(l-xOffset)
        b = str(b-yOffset)
        r = str(r-xOffset)
        t = str(t-yOffset)

        tid = dic[image][tile]

        mbr = '0'*(4-len(nver))+ nver + ',' + '0'*(4-len(l))+ l + ' ' + \
                '0'*(4-len(r))+ r + ' ' + '0'*(4-len(b))+ b + ' ' +  \
                '0'*(4-len(t))+ t

        shape = ','.join(['0'*(4-len(x))+ x + ' ' + '0'*(4-len(y))+y for (x, y) in shp])

        print str(tid)+ tab + ','.join((str(did), str(oid),mbr, shape))

    sys.stdout.flush()

if __name__ == '__main__':
    main()

