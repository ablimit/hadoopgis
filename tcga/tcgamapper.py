#! /usr/bin/python

import sys
import os

def main():
    i = 1
    fullfilename = os.environ['map_input_file']
    sys.stderr.write(fullfilename+ "\n")
    if fullfilename.endswith('.2.tsv'):
        tag = "2"
    else:
        tag = "1"

    for line in sys.stdin:
        arr = line.strip().split("\t")	
        print "\t".join((arr[0], tag, "\t".join(arr[1:])))  

    sys.stdout.flush()

if __name__ == '__main__':
    main()
