#! /bin/bash

loc=/data2/ablimit/Data/spatialdata
make -f Makefile

echo "started building rtree"

./rtreeloader ${loc}/osmout/osm_polygon_planet.dat ${loc}/osm/rtreeindex 1000 5000 0.9

echo "done."
