#! /usr/bin/python

import sys
from collections import defaultdict

def simpMBR(coor):
    sp = coor.split(" ")
    l = int(sp[0]) - 49152
    r = int(sp[1]) - 49152
    b = int(sp[2]) - 4096
    t = int(sp[3]) - 4096
    return ' '.join([str(item) for item in (l,r,b,t)])

def simpPoint(coor):
    sp = coor.split(" ")
    return ' '.join((str(int(sp[0])-49152),str(int(sp[1])-4096)))

def main():
    X = 49152
    Y = 4096
    for line in sys.stdin:
        sp = line.strip().split(",")
        sp[3] = simpMBR(sp[3])
        for i in range(4,len(sp)):
            sp[i] = simpPoint(sp[i])

        print ','.join(sp)

    sys.stdout.flush()

if __name__ == '__main__':
  main()

