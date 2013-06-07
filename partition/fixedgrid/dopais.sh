#! /bin/bash

datapath=""

for size in 256 512 768 1024 2048 4096 8192 16384
do
    mkdir grid${size}

    for f in astroII.1 astroII.2 gbm0.1 gbm0.2 gbm1.1 gbm1.2 gbm2.1 gbm2.2 normal.2 normal.3 oligoastroII.1 oligoastroII.2 oligoastroIII.1 oligoastroIII.2 oligoII.1 oligoII.2 oligoIII.1 oligoIII.2
    do
        echo "Histograms for $f: "

        ./mapper  -w 0 -s 0 -n 57344 -e 110592 -x ${xsplit} -y ${ysplit} < ../data/algo1${f}.markup.ablet.1 > out.txt 2> err.txt
        cat data/algo1/${f}.markup.ablet.1 data/algo2/${f}.markup.ablet.2 | genmbb | python data/normalize.py pais > /dev/shm/mbb.txt

        lc=`wc -l /dev/shm/mbb.txt | cut -d' ' -f1 `
        p=`expr $((lc/10000))`
    done
done

