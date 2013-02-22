#! /bin/bash

psql --dbname=osmeu --no-align --pset=footer=off --command="SELECT id, tilename, osm_id FROM planet_osm_polygon ;" --output=/data2/ablimit/Data/spatialdata/osmout/europe_osm_tileid.dat

psql --dbname=osm   --no-align --pset=footer=off --command="SELECT id, tilename, osm_id FROM planet_osm_polygon ;" --output=/data2/ablimit/Data/spatialdata/osmout/planet_osm_tileid.dat

