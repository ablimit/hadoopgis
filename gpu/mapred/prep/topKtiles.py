#! /usr/bin/python

import sys

def main():
    if len(sys.argv) < 2:
        sys.stderr.write("Usage: "+ sys.argv[0] + "\n")
        exit(1)

    k = int(sys.argv[1])
    tab  = "\t"
    i = 0 
    previd = ""
    for line in sys.stdin:
        tid  = line.strip().split(tab)[0]
        if tid != previd:
            i +=1
            if i > k:
                break
        sys.stdout.write(line)
        previd = tid
    sys.stdout.flush()

if __name__ == '__main__':
    main()

