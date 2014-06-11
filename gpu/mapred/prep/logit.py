#! /usr/bin/python

import sys
import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def residual(obja,objb,vera,verb,speedup):
    res =  speedup - (4.993 - 9.722 * 0.00001 *obja + 2.455 * 0.001 * objb - 2.604 * 0.000001*vera - 2.434 * 0.00001* verb)
    return res 

def main():
    for line in sys.stdin:
        sp = line.strip().split(",")
        print residual(float(sp[0]), float(sp[1]), float(sp[2]), float(sp[3]),
                       float(sp[4]))
    sys.stdout.flush()

if __name__ == '__main__':
    main()

