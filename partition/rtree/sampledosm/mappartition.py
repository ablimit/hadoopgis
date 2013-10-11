#! /home/aaji/softs/bin/python

import sys
import math

dic = {}

def main():

    if len(sys.argv) < 2:
        sys.stderr.write("Usage: "+ sys.argv[0] + "\n")

    for line in open(sys.argv[1],'r'):
        sp =line.strip().split()
	pid = int(sp[0])
	dic [pid] = set([int(item) for item in sp[1:]])

    for line in sys.stdin:
        sp =line.strip().split()
        if len(sp)>1:
            oid = int(sp[0])
	    # pid | oid | shape 
	    pids = []
	    
	    for pid in dic:
		if oid in dic[pid]:
		    pids.append(pid)
	    
	    for pid in pids:
		print '|'.join((str(pid),sp[0],' '.join(sp[1:-1])))

    sys.stdout.flush()


if __name__ == '__main__':
    main()

