#! /usr/bin/python

import sys
import string
import os


def main():

    first = True
    print >>sys.stdout, '|'.join(('PAIS_UID','SEQNUM','TILENAME','MARKUP_ID','POLYGON'))
    for line in sys.stdin:
	if first:
	    first =False
	    continue 
	fields = line.strip().split("|")
	if fields[0].endswith("1"):
	    seqnum = '1'
	else:
	    seqnum = '2'
	print >>sys.stdout, '|'.join((fields[0],seqnum,'|'.join(fields[1:])))
    sys.stdout.flush()

# main function 
main()

