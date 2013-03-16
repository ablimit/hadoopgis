#! /bin/bash

loc=/data2/ablimit/Data/spatialdata
make -f Makefile

# echo "generating mbb"
# ./preprocess <${loc}/osmout/planet.dat.1 > ${loc}/osmout/osm.mbb.dat

echo "started building rtree"
./rtreeloader /scratch/ablimit/osm/osm.mbb.dat /scratch/ablimit/osm/index/planet 1000 5000 0.9

echo "done."
