#! /usr/bin/python
import sys

def main():
    xmax = 110592.0
    ymax = 57344.0
    for line in sys.stdin:
        line =line.strip()
        sp = line.split("|")
        aid = int(sp[2])
        coors = [ float(x) for x in sp[4].split()]
        coors[0] = coors[0]/xmax
        coors[2] = coors[2]/xmax
        coors[1] = coors[1]/ymax
        coors[3] = coors[3]/ymax
        coors= [str(x) for x in coors]
        coors.insert(0,sp[3])

    if 1 == aid:
        sys.stdout.write(" ".join(coors) + '\n')
    else:
        sys.stderr.write(" ".join(coors) + '\n')

    sys.stdout.flush()
    sys.stderr.flush()

if __name__ == '__main__':
    main()


