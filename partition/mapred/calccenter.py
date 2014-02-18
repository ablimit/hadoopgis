#! /usr/bin/python

import sys


def main():
    for line in sys.stdin:
        fields = line.strip().split()
        if len(fields) == 5:
            x = (float(fields[1]) +float(fields[3]))/2
            y = (float(fields[2]) +float(fields[4]))/2
            print ' '.join((' '.join(fields),' '.join((str(x),str(y)))))
    sys.stdout.flush()

if __name__ == '__main__':
    main()

