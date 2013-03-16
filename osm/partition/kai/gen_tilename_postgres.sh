#! /bin/bash


psql -d osm -c "BEGIN; ALTER TABLE planet_osm_polygon ADD COLUMN tilename8x8 varchar(32); UPDATE planet_osm_polygon SET tilename8x8 = tile8x8.tilename FROM tile8x8 WHERE ST_Contains(tile8x8.mbb,planet_osm_polygon.way) = TRUE;  COMMIT; "

