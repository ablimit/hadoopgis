#! /bin/bash

for f in 01 05 10 15 20 25 
do
  echo "sampling ${f}%"
  cat  /home/aaji/proj/data/osm/osm.mbb.norm.filter.dat | awk "BEGIN {srand()} !/^$/ { if (rand() <= .${f}) print \$0}" > /home/aaji/proj/data/sampling/osm/osm.sample.${f}.dat
done

