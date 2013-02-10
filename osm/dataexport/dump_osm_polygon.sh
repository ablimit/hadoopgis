#! /bin/bash

# -- CREATE TABLE clean_polygon (osm_id  bigint, z_order real, area  real, );
# -- SELECT AddGeometryColumn('clean_polygon','way', 4326, 'POLYGON', 2 ) ;

psql --no-align --pset=footer=off --command="SELECT id, tilename, osm_id, z_order, ST_AsText(way) AS way FROM clean_osm ;" | gzip > /dev/shm/planet_osm_polygon_clean.dat.gz
# psql --no-align --pset=footer=off --command="SELECT id, tilename, osm_id, z_order, ST_AsText(way) AS way FROM clean_osm ;" | gzip > /data2/ablimit/Data/spatialdata/osmout/planet_osm_polygon_clean.dat.gz
