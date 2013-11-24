#! /usr/bin/python

import sys
import math


def main():
  
  for line in sys.stdin:
    sp =line.strip().split()
    print ' '.join(("set object rect from", sp[1] + "," + sp[2], "to", sp[3] + "," + sp[4]))
  
  sys.stdout.flush()


if __name__ == '__main__':
  main()

