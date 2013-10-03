#! /home/aaji/softs/bin/python

import sys
import re
import math

def main():
    if len(sys.argv) <2:
        print "Usage: "+ sys.argv[0] + " [sid]"
        sys.exit(1)
    sid = sys.argv[1].strip()

    for line in sys.stdin:
        sp = line.strip().split("|")
	print "|".join((sp[0],sid,sid,str(int(sp[1])),sp[-1]))
    sys.stdout.flush()
    sys.stderr.flush()

if __name__ == '__main__':
    main()

