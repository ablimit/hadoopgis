#! /usr/bin/python

import sys
import string
import os


def main():

    if (len(sys.argv)<3):
	sys.stderr.write("Missing params.\n")
	sys.exit()
    path   = sys.argv[1]
    sequence = sys.argv[2]
    
    if not path.endswith('/'):
	path = path + '/'


    #  print >>sys.stdout, '|'.join(('PAIS_UID','TILENAME','MARKUP_ID','POLYGON'))
    for img in os.listdir(path):
	f = open(path+img,'r')
	for line in f:
	    fields = line.strip().split(",",2)
	    polygon = fields[-1]
	    #fields[0] = fields[0].replace('"','')
	    #fields[1] =fields[1].replace('"','')
	    polygon = "POLYGON(( " + polygon[1:-1] + "))"
	    print >>sys.stdout, '|'.join((img,fields[0],sequence,polygon))
	sys.stdout.flush()
	f.close()
    
# main function 
main()

