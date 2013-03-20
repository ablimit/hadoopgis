#! /home/aaji/softs/bin/python

import sys

dumvar ="-1"
bar = "|" 

for line in sys.stdin:
    sp =line.strip().split("\t",3)
    print bar.join((sp[2],sp[1],dumvar,sp[0],sp[3]))

sys.stdout.flush()

