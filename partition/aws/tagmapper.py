#! /usr/bin/python

import sys
import os

def main():

    if len(sys.argv) < 2:
        sys.stderr.write("Usage: "+ sys.argv[0] + "\n")
        sys.exit(1)

    i = 1
    fullfilename = os.environ['map_input_file']
    sys.stderr.write(fullfilename+ "\n")
    tag = -1
#    print str(sys.argv)
    for t in sys.argv[1:]:
	#print str(t)
	#print "\n"
	if fullfilename.find(t) > -1:
		tag = i
		break
	i += 1	
    
    for line in sys.stdin:
        arr = line.strip().split("\t")	
 	print "\t".join((arr[0], str(tag), "\t".join(arr[1:])))  
 
    sys.stdout.flush()

if __name__ == '__main__':
    main()
