#! /bin/bash

tilecolumn=tilenameb

psql --dbname=osm --no-align --pset=footer=off --command="SELECT id, ${tilecolumn}, osm_id, z_order, ST_AsText(way) AS way FROM planet_osm_polygon ;" --output=/data2/ablimit/Data/spatialdata/osmout/osm_polygon_planet.dat

psql --dbname=osmeu --no-align --pset=footer=off --command="SELECT id, ${tilecolumn}, osm_id, z_order, ST_AsText(way) AS way FROM planet_osm_polygon ;" --output=/data2/ablimit/Data/spatialdata/osmout/osm_polygon_europe.dat

