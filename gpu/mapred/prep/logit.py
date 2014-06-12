#! /usr/bin/python

import sys
import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def reg(obja,objb,vera,verb,speedup):
    res =  4.993 - 9.722 * 0.00001 *obja + 2.455 * 0.001 * objb - 2.604 * 0.000001*vera - 2.434 * 0.00001* verb
    # res =  7.72681 - 18.07326 * obja  + 391.38156 * objb - 27.72306 * vera -243.30210 * verb
    return res 

def main():
    for line in sys.stdin:
        sp = line.strip().split(",")
        print reg(float(sp[0]), float(sp[1]), float(sp[2]), float(sp[3]),
                       float(sp[4]))
    sys.stdout.flush()

if __name__ == '__main__':
    main()

