#! /home/aaji/softs/bin/python

import sys

dumvar =-1 
bar = "|" 

for line in sys.stdin:
    sp =line.strip().split("\t",2)
    if (sp[1].count('_')>1):
	print line.strip()

sys.stdout.flush()

