#! /bin/bash


for f in astroII.1 astroII.2 gbm0.1 gbm0.2 gbm1.1 gbm1.2 gbm2.1 gbm2.2 normal.2 normal.3 oligoastroII.1 oligoastroII.2 oligoastroIII.1 oligoastroIII.2 oligoII.1 oligoII.2 oligoIII.1 oligoIII.2
do
    for method in rtree minskew rv rkHist sthist 
    do
        if [ -e partres/${f}.${method}.txt ]
        then
            echo "Evaluating ${method} for image ${f}."
            python evalpartition.py partres/${f}.${method}.txt < pais/${f}.mbb.txt > parteval/${f}.${method}.dat
        else
            echo "missing ${f}.${method}.txt"
        fi
    done

done



