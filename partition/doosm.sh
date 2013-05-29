#! /bin/bash

f=osm.mbb.norm.filter

echo "$f"
lc=`wc -l data/osm/${f}.dat | cut -d' ' -f1 `
p=`expr $((lc/5000000))`

#rtree.sh
./rtree.sh $p data/osm/${f}.dat data/partres/${f}.rtree.txt
#minskew.sh
./minskew.sh $p data/osm/${f}.dat data/partres/${f}.minskew.txt I 10
#rkHist.sh
./rkHist.sh $p data/osm/${f}.dat data/partres/${f}.rkHist.txt 0.1
#rv.sh
./rv.sh $p data/osm/${f}.dat data/partres/${f}.rv.txt 0.4 
#stHist.sh
./stHist.sh $p data/osm/${f}.dat data/partres/${f}.sthist.txt 0.5
