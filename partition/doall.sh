#! /bin/bash

jvm="-Xss1m -Xmx3500M"

for f in astroII.1 astroII.2 gbm0.1 gbm0.2 gbm1.1 gbm1.2 gbm2.1 gbm2.2 normal.2 normal.3 oligoastroII.1 oligoastroII.2 oligoastroIII.1 oligoastroIII.2 oligoII.1 oligoII.2 oligoIII.1 oligoIII.2
do
    echo "Histograms for $f: "
    cat data/algo1/${f}.markup.ablet.1 data/algo2/${f}.markup.ablet.2 | genmbb > /dev/shm/mbb.txt
    
    lc=`wc -l /dev/shm/mbb.txt | cut -d' ' -f1 `
    p=`expr $((lc/10000))`

    #rtree.sh
    echo  "RTree  "
    ./rtree.sh "${jvm}" $p /dev/shm/mbb.txt data/partres/pais/${f}.rtree.txt
    #minskew.sh
    echo  "MinSkew  "
    ./minskew.sh "${jvm}" $p /dev/shm/mbb.txt data/partres/pais/${f}.minskew.txt I 8
    #rkHist.sh
    echo  "rkHist  "
    ./rkHist.sh "${jvm}" $p /dev/shm/mbb.txt data/partres/pais/${f}.rkHist.txt 0.1
    #rv.sh
    echo  "RV  "
    ./rv.sh "${jvm}" $p /dev/shm/mbb.txt data/partres/pais/${f}.rv.txt 0.4 
    #stHist.sh
    echo "stHist"
    ./stHist.sh "${jvm}" $p /dev/shm/mbb.txt data/partres/pais/${f}.sthist.txt 0.5
done

#rv.sh
#stHist.sh

