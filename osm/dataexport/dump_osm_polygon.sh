#! /bin/bash

psql --no-align --pset=footer=off --command="SELECT id, tilename, osm_id, z_order, ST_AsText(way) AS way FROM clean_osm ;" | gzip > /dev/shm/planet_osm_polygon_clean.dat.gz
