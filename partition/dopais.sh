#! /bin/bash

jvm="-Xss4M -Xmx20000M"
datapath=/mnt/scratch1/aaji

if [ ! -e genpaismbb ] ;
then 
    echo "MBR calculator [genpaismbb] is missing."
    exit 0;
fi

size=10000
mark="oc${size}"

if [ ! -e data/partres/pais/${mark} ] ;
then 
    mkdir -p data/partres/pais/${mark}
fi

for f in astroII.1 astroII.2 gbm0.1 gbm0.2 gbm1.1 gbm1.2 gbm2.1 gbm2.2 normal.2 normal.3 oligoastroII.1 oligoastroII.2 oligoastroIII.1 oligoastroIII.2 oligoII.1 oligoII.2 oligoIII.1 oligoIII.2
do
    echo "Histograms for $f: "
    cat ${datapath}/algo1/${f}.markup.ablet.1 ${datapath}/algo2/${f}.markup.ablet.2  | genpaismbb | python data/normalize.py pais > /dev/shm/paismbb.txt

    lc=`wc -l /dev/shm/paismbb.txt | cut -d' ' -f1 `
    p=`expr $((lc/size))`
    
    #echo "${p}"
    # exit 0;

    #rtree.sh
    echo  "RTree  "
    ./rtree.sh "${jvm}" $p /dev/shm/paismbb.txt data/partres/pais/${mark}/${f}.rtree.txt
    #minskew.sh
    echo  "MinSkew  "
    ./minskew.sh "${jvm}" $p /dev/shm/paismbb.txt data/partres/pais/${mark}/${f}.minskew.txt I 7
    #rkHist.sh
    echo  "rkHist  "
    ./rkHist.sh "${jvm}" $p /dev/shm/paismbb.txt data/partres/pais/${mark}/${f}.rkHist.txt 0.1
    #rv.sh
    echo  "RV  "
    ./rv.sh "${jvm}" $p /dev/shm/paismbb.txt data/partres/pais/${mark}/${f}.rv.txt 0.4 
    #stHist.sh
    echo "stHist"
    ./stHist.sh "${jvm}" $p /dev/shm/paismbb.txt data/partres/pais/${mark}/${f}.sthist.txt 0.5
done

