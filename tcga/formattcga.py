#! /usr/bin/python

import sys

def main():
  if len(sys.argv) <2:
    print "Usage: "+ sys.argv[0] + " [partition file]"
    sys.exit(1)

  filename = sys.argv[1]
  if "/" in sys.argv[1]:
    filename = sys.argv[1].split("/")[-1]
  prefix = "-".join(filename.split('.',1)[0].split("-")[1:])

  with open(sys.argv[1],'r') as f:
    for i,line in enumerate(f):
      sp = line.strip().split(",",2)
      tid = "-".join((sp[0].lower(),sp[1]))
      oid = "-".join((prefix,str(i)))
      obj = "".join(("POLYGON((",sp[-1][1:-1],"))"))
      print "\t".join((tid,oid,obj))

  sys.stdout.flush()
  sys.stderr.flush()
# sys.exit(0)

if __name__ == '__main__':
  main()

