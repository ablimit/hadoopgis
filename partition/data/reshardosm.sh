#! /bin/bash

# file path
path=/data2/ablimit/Data/spatialdata/osmout
f1=${path}/planet.dat.1
f2=${path}/europe.dat.2

echo "Generating MBRs for planet"
genmbb 1 < ${f1} | python normalize.py osm > /dev/shm/mbb.planet.txt

echo "Generating MBRs for europe"
genmbb 2 < ${f2} | python normalize.py osm > /dev/shm/mbb.europe.txt


echo "Re-Partition the map"
for method in rtree minskew rv rkHist sthist 
do
    if [ -e partres/osm/${method}.txt ]
    then
        echo "resharding for method ${method} .."
        cat /dev/shm/mbb.planet.txt /dev/shm/mbb.europe.txt | python genpid.py partres/${f}.${method}.txt | python reshardosm.py ${f1} ${f2} | bzip2 > repart/osm/${method}.bz2
    fi
done

