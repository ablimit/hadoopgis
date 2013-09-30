#! /usr/bin/python

""" This program transforms pais dataset into a similar format of OSM dataset. 
    Specifically, the folders which need to be processed will be passed from command line,
    then the program will generate data in following fromat:
    pid | image-name | oid | sid | geometry 
"""

import os
import sys
import math
from os import walk

def main():
    
    oid = 0
    
    for file_path in sys.argv[1:]:

	# check if the file is a directory 
	file_path = file_path.strip()
	if not os.path.isdir(file_path):
	    sys.stderr.write(file_path + ' is NOT a valid directory.\n')
	    sys.exit(0)
	
	# check if the file is algo1 or algo2
	sid = file_path[-1]
	if (sid != "1" and sid != "2"):
	    sys.stderr.write('only algorithms 1 or 2 is allowed.\n')
	    sys.exit(0)
	
	sys.stderr.write('dir: '+ file_path+ '\n')
	
	#recursivly visit files
	for (dirpath, dirnames, filenames) in walk(file_path):
	    for f in filenames:
		sys.stderr.write('\tfile: '+f+'\n')
		for line in open(dirpath+'/'+f,'r'):
		    line = line.strip()
		    sp = line.split('|')
		    tile = sp[0].split("-",1)
		    image_name = tile[0]
		    pid = tile[1]
		    oid += 1
		    geom = sp[-1]

		    print "|".join((pid,image_name,sid,str(oid),geom))

    sys.stdout.flush()


if __name__ == '__main__':
    main()
