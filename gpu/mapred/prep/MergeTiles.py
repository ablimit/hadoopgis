#! /usr/bin/python

import sys
from collections import defaultdict

def readITMap2(path):
    image_tile_map = defaultdict(dict)
    tid = 1
    for line in open(path,'r'):
        sp = line.strip().split('|')
        image = sp[1]
        tile  = sp[0]
        image_tile_map[image][tile] = tid
        tid += 1
    return image_tile_map

def readITMap(path):
    tid = 1
    image_tile_map = defaultdict(dict)
    images = ('astroII.1', 'astroII.2', 'gbm0.1', 'gbm0.2', 'gbm1.1', 'gbm1.2',
               'gbm2.1', 'gbm2.2', 'normal.2', 'normal.3', 'oligoastroII.1',
               'oligoastroII.2', 'oligoastroIII.1', 'oligoastroIII.2',
               'oligoII.1', 'oligoII.2', 'oligoIII.1', 'oligoIII.2')
    for imageId  in images:
        for x in xrange(0,110592,4096):
            for y in xrange(0,57344,4096):
                sx = str(x)
                sy = str(y)
                tileId = '-'.join(('0'*(10-len(sx)) + sx, '0'*(10-len(sy))+ sy ))
                image_tile_map[imageId][tileId] = tid
                tid += 1
    return image_tile_map


def getOffSets(tileId, xPixels=8192, yPixels=8192):
    sp = tileId.split('-')
    x = int(sp[0])
    y = int(sp[1])
    x = (x/xPixels) * xPixels
    y = (y/yPixels) * yPixels
    return x,y

def getParentTile(tileId):
    x,y = getOffSets(tileId)
    x = str(x)
    y = str(y)
    return '-'.join(('0'*(10-len(x)) + x, '0'*(10-len(y)) + y ))

def main():
    # print getOffSets(sys.argv[1].strip(),8192,8192)
    # print getParentTile(sys.argv[1].strip())
    # return
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
        tileId = sp[0]
        xOffset, yOffset = getOffSets(tileId)

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

        tid = dic[image][getParentTile(tileId)]
        sys.stderr.write('|'.join((image , tileId, '\n'))

        mbr = '0'*(4-len(nver))+ nver + ',' + '0'*(4-len(l))+ l + ' ' + \
                '0'*(4-len(r))+ r + ' ' + '0'*(4-len(b))+ b + ' ' +  \
                '0'*(4-len(t))+ t

        shape = ','.join(['0'*(4-len(x))+ x + ' ' + '0'*(4-len(y))+y for (x, y) in shp])

        print str(tid)+ tab + ','.join((str(did), str(oid),mbr, shape))

    sys.stdout.flush()

if __name__ == '__main__':
    main()

