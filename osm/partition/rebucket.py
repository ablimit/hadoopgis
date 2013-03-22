#! /home/aaji/softs/bin/python

import sys

dumvar ="-1"
bar = "|" 

broken_record=0 

for line in sys.stdin:
    sp =line.strip().split("\t",3)
    if (len(sp)==4):
	print bar.join((sp[2],sp[1],dumvar,sp[0],sp[3]))
    else:
	broken_record += 1 
	sys.stderr.write(str(broken_record)+"\n")


sys.stdout.flush()

