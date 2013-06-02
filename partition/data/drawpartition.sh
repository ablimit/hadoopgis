#! /bin/bash

tempf=/tmp/zahide.gnu

for f in astroII.1 
    # astroII.2 gbm0.1 gbm0.2 gbm1.1 gbm1.2 gbm2.1 gbm2.2 normal.2 normal.3 oligoastroII.1 oligoastroII.2 oligoastroIII.1 oligoastroIII.2 oligoII.1 oligoII.2 oligoIII.1 oligoIII.2
do
    echo "drawing ${f} .."
    for method in rtree minskew rv rkHist sthist 
    do
        if [ -e partres/${f}.${method}.txt ]
        then
            python drawSinglePartition.py pics/${f}.${method}.png < partres/${f}.${method}.txt >> ${tempf}
            gnuplot ${tempf}
        fi

    done

done

