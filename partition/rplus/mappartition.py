#! /usr/bin/python
import sys

dic = {}

def readstdin():
  global dic
  for line in sys.stdin:
    line = line.strip()
    sp =line.split()
    if len(sp)>1:
      dic[int(sp[0])] = line

def main():
  global dic
  if len(sys.argv) < 2:
    sys.stderr.write("Usage: "+ sys.argv[0] + "\n")
  
  readstdin()
  
  for line in open(sys.argv[1],'r'):
    sp =line.strip().split()
    pid = sp[0]
    for oid in set([int(item) for item in sp[1:]]):
      print " ".join((pid,dic[oid]))
  
  sys.stdout.flush()

if __name__ == '__main__':
  main()

