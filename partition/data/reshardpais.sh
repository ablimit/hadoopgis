#! /bin/bash


for f in astroII.1 astroII.2 gbm0.1 gbm0.2 gbm1.1 gbm1.2 gbm2.1 gbm2.2 normal.2 normal.3 oligoastroII.1 oligoastroII.2 oligoastroIII.1 oligoastroIII.2 oligoII.1
    #oligoII.2 oligoIII.1 oligoIII.2
do
    echo "$f"
    for method in minskew rv rkHist sthist 
    do
        echo ${method}
        python genpid.py partres/${f}.${method}.txt < pais/${f}.mbb.txt 2> partres/${f}.${method}.info | python reshardPAIS.py algo1/${f}.markup.ablet.1 > algo1/${f}.${method}.txt
    done


done

