#! /usr/bin/python

import sys
import string
import os

# oligoII.1-0000081920-0000008192

def main():

#    if (len(sys.argv)<3):
#	sys.stderr.write("Missing params.\n")
#	sys.exit()
    maxx = -1
    maxy = -1

    lines = sys.stdin.readlines()
    for line in lines:
	fields = line.strip().split('-',2)
	x = int(fields[1])
	y = int(fields[2])
	if x > maxx:
	    maxx = x

	if y > maxy:
	    maxy = y

    print "max X coordinate", maxx
    print "max Y coordinate", maxy

    x0 = maxx/4
    y0 = maxy/4
    
    x1 = x0 + maxx/2
    y1 = y0
    
    x2=  x1
    y2 = y0 +maxy/2

    x3 = x0
    y3 = y2
    space = ' '
    print ','.join((
	    space.join((str(x0),str(y0))),
	    space.join((str(x1),str(y1))),
	    space.join((str(x2),str(y2))),
	    space.join((str(x3),str(y3))),
	    space.join((str(x0),str(y0)))))
    
# main function 
main()


