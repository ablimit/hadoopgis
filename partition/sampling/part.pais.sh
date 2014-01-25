#! /bin/bash

for image in astroII.1 astroII.2 gbm0.1 gbm0.2 gbm1.1 gbm1.2 gbm2.1 gbm2.2 normal.2 normal.3 oligoastroII.1 oligoastroII.2 oligoastroIII.1 oligoastroIII.2 oligoII.1 oligoII.2 oligoIII.1 oligoIII.2
do
    echo "sampling ${image}"
  for f in 01 05 10 15 20 25 
  do
    cat  /home/aaji/proj/data/pais/mbb/${image}.norm.1.dat | awk "BEGIN {srand()} !/^$/ { if (rand() <= .${f}) print \$0}" > /home/aaji/proj/data/sampling/pais/${image}.sample.${f}.dat
  done
done

