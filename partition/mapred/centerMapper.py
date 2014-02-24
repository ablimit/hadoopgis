#! /usr/bin/python

import sys

def main():
    cc = 0 
    for line in sys.stdin:
        fields = line.strip().split()
        if len(fields) == 5:
            x = (float(fields[1]) +float(fields[3]))/2
            y = (float(fields[2]) +float(fields[4]))/2
            cc +=1
            sys.stdout.write('\t'.join((','.join((str(x),str(y))),line)))
    sys.stdout.flush()
    sys.stderr.write(str(cc)+"\n")

if __name__ == '__main__':
    main()

