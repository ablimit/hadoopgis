#! /usr/bin/python

import sys
import random


def main():
  frac = float(sys.argv[1])
  geoms = []
  for line in sys.stdin:
      geoms.append(line.strip())

  for item in random.sample(geoms, len(geoms)*frac):
      print item
  
  sys.stdout.flush()


if __name__ == '__main__':
  main()

