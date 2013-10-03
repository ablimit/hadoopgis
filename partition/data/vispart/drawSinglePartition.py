#! /usr/bin/python

import sys
import math

def main():
    if len(sys.argv) <2:
        print "Usage: "+ sys.argv[0] + " [output image path]"
        sys.exit(1)
    
    opath = ""
    objpath = ""
    partpath = ""

    for s in sys.argv[1:]:
        if "png" in s:
            opath = s
        elif "obj" in s:
            objpath = s
        elif "part" in s:
            partpath = s
        else:
            sys.stderr.write("Weird argument !\n")
            sys.exit(0)


    print "set terminal pngcairo size 1024,768 enhanced font 'Verdana,20'"
    print "set output '{0}'".format(opath)
    print "unset xtics" 
    print "unset ytics"

    oid = 0

    for line in open(objpath,"r"):
        sp = line.strip().split()
        if (len(sp)>4):
            oid += 1
            x1 = sp[1]
            y1 = sp[2]
            x2 = sp[3]
            y2 = sp[4]
            print "set object {0} rect from {1}, {2} to {3}, {4}".format(oid,x1,y1,x2,y2)
    
    for line in open(partpath,"r"):
        sp = line.strip().split()
        if (len(sp)>4):
            oid += 1
            x1 = sp[1]
            y1 = sp[2]
            x2 = sp[3]
            y2 = sp[4]

            # default color
            col = ""
            color_id = oid % 7

            if color_id == 0:
                col = "red"
            elif color_id ==1:
                col = "orange"
            elif color_id ==2:
                col = "yellow"
            elif color_id ==3:
                col = "green"
            elif color_id ==4:
                col = "cyan"
            elif color_id ==5:
                col = "blue"
            elif color_id ==6:
                col = "violet"
            else:
                col = "black"

            print 'set object {0} rect from {1}, {2} to {3}, {4} fs empty border rgb "{5}" lw 3'.format(oid,x1,y1,x2,y2,col)

    
    
    print "plot [-0.05:1.05] [-0.05:1.05] NaN notitle "

    sys.stdout.flush()
    sys.stderr.flush()

if __name__ == '__main__':
    main()

