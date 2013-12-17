#! /usr/bin/python

""" 
    This program matches markup files and mbb files.
"""

import os
import sys

def main():
    if len(sys.argv) < 2:
        sys.stderr.write("error: Missing param. [markup file]")
        sys.exit(1)

    geomfile = []
    mbbfile = []

    with open(sys.argv[1],'r') as infile:
        for line in infile:
            sp = line.strip().split('|')
            geomfile.append(sp[-1])

    for line in sys.stdin:
        mbbfile.append(line.strip())
    
    for (i, line) in enumerate(mbbfile):
        sp = line.split('|')
        sp[-1] = geomfile[i]
        print "|".join(sp)

    sys.stdout.flush()

if __name__ == '__main__':
    main()

