#! /bin/bash


psql --dbname=osm --no-align --pset=footer=off --command="SELECT tilename, x,y,width,height, ST_AsText(mbb) AS boundary FROM tileb ;" --output=/data2/ablimit/Data/spatialdata/osmout/osm_tileb.dat
