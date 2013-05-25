#! /bin/bash


for f in astroII.1 astroII.2 gbm0.1 gbm0.2 gbm1.1 gbm1.2 gbm2.1 gbm2.2 normal.2 normal.3 oligoastroII.1 oligoastroII.2 oligoastroIII.1 oligoastroIII.2 oligoII.1 oligoII.2 oligoIII.1 oligoIII.2
do
    echo "$f"
    #genmbb < algo1/${f}.markup.ablet.1 > pais/${f}.mbb.txt
    mv pais/${f}.mbb.txt pais/${f}.mbb.txt.tmp
    python normolize.py pais < pais/${f}.mbb.txt.tmp > pais/${f}.mbb.txt
done

