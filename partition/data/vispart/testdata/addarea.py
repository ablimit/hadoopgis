#! /home/aaji/softs/bin/python

import sys
import math

# from collections import defaultdict
# dic = defaultdict(set)


def main():
    for line in sys.stdin:
	sp =line.strip().split()
	if len(sp)>4:
	    area = (float(sp[4])-float(sp[2]) ) * (float(sp[3])-float(sp[1]) )
	    sp.append(str(area))
	    print '\t'.join(sp)

    sys.stdout.flush()


if __name__ == '__main__':
    main()

