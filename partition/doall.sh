#! /bin/bash


for f in astroII.1 astroII.2 gbm0.1 gbm0.2 gbm1.1 gbm1.2 gbm2.1 gbm2.2 normal.2 normal.3 oligoastroII.1 oligoastroII.2 oligoastroIII.1 oligoastroIII.2 oligoII.1 oligoII.2 oligoIII.1 oligoIII.2
do
    echo "$f"
    lc=`wc -l data/pais/${f}.mbb.txt | cut -d' ' -f1 `
    p=`expr $((lc/5000))`

    #rtree.sh
    ./rtree.sh $p data/pais/${f}.mbb.txt data/partres/${f}.rtree.txt
    #minskew.sh
    ./minskew.sh $p data/pais/${f}.mbb.txt data/partres/${f}.minskew.txt I 8
    #rkHist.sh
    ./rkHist.sh $p data/pais/${f}.mbb.txt data/partres/${f}.rkHist.txt 0.1
    #rv.sh
    ./rv.sh $p data/pais/${f}.mbb.txt data/partres/${f}.rv.txt 0.4 
    #stHist.sh
    ./stHist.sh $p data/pais/${f}.mbb.txt data/partres/${f}.sthist.txt 0.9
done

#rv.sh
#stHist.sh

