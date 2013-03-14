#! /usr/bin/python

import sys

mapping ={} 
OSM_ID =1
for line in open(sys.argv[1],"r"):
    sp =line.strip().split("\t")
    mapping[sp[0]]=sp[1]

for line in open(sys.argv[2],"r"):
    sp =line.strip().split("|",4)
    if sp[OSM_ID] in mapping:
	sp[OSM_ID] = mapping[sp[OSM_ID]]
    else:
	sp[OSM_ID] = "NULL"

    print "|".join(sp)

