#! /bin/bash

psql --dbname=osm --no-align --pset=footer=off --command="SELECT id, tilenameb, osm_id, z_order, ST_AsText(way) AS way FROM planet_osm_polygon ;" --output=/data2/ablimit/Data/spatialdata/osmout/planet_osm_polygon.dat

psql --dbname=osmeu --no-align --pset=footer=off --command="SELECT id, tilenameb, osm_id, z_order, ST_AsText(way) AS way FROM planet_osm_polygon ;" --output=/data2/ablimit/Data/spatialdata/osmout/europe_osm_polygon.dat

