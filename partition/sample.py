#! /usr/bin/python

import sys
import random


def main():
  frac = float(sys.argv[1])
  
  for line in sys.stdin:
    if random.random() > frac:
      print line.strip()
  
  sys.stdout.flush()


if __name__ == '__main__':
  main()

