#! /bin/bash

loc=/data2/ablimit/Data/spatialdata
make -f Makefile

echo "generating mbb"
./preprocess <${loc}/osmout/planet.dat.1 > ${loc}/osmout/osm.mbb.dat

echo "started building rtree"
#./rtreeloader ${loc}/osmout/planet.dat.1 ${loc}/osm/rindex/planet.idx 1000 5000 0.9

echo "done."
