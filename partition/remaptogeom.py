#! /usr/bin/python
import sys
import gzip


def readgeom(path):
    dic = {}
    for line in gzip.open(path,'r'):
        sp= line.strip().split("|")
        if len(sp)>1:
            dic[sp[0]] = sp[-1]
    return dic

def main():
    if len(sys.argv) < 2:
        sys.stderr.write("Usage: "+ sys.argv[0] + "\n")
        exit(1)
    path = sys.argv[1]
    if not path.endswith('.gz'):
        sys.stderr.write(path +" is not a gzipped file.\n")
        exit(1)

    dict = readgeom(path)
    for line in sys.stdin:
        sp =line.strip().split()
        pid = sp[0]
        uid = sp[1]
        print "\t".join((pid,uid,dict[uid]))
  
  sys.stdout.flush()

if __name__ == '__main__':
  main()

