#! /bin/bash

tempf=/tmp/zahide.gnu

for f in astroII.1 astroII.2 gbm0.1 gbm0.2 gbm1.1 gbm1.2 gbm2.1 gbm2.2 normal.2 normal.3 oligoastroII.1 oligoastroII.2 oligoastroIII.1 oligoastroIII.2 oligoII.1 oligoII.2 oligoIII.1 oligoIII.2
do
    echo "drawing ${f} .."
    for method in minskew rv rkHist sthist 
    do
        if [ -e partres/${f}.${method}.txt ]
        then
            echo "set terminal pngcairo size 1024,768 enhanced font 'Verdana,20'" > ${tempf}
            echo "set output 'pics/${f}.${method}.png'" >> ${tempf}
            echo "unset xtics" >> ${tempf}
            echo "unset ytics" >> ${tempf}

            cat partres/${f}.${method}.txt | while read oid x y xx yy;
        do
            oid=`expr $oid + 1`
            echo "set object ${oid} rect from ${x}, ${y} to ${xx}, ${yy}" >> ${tempf}
        done
        echo "plot [-0.05:1.05] [-0.05:1.05] NaN notitle "  >> ${tempf}
        gnuplot ${tempf}
    fi

done


done

